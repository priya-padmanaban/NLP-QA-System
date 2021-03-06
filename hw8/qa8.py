import re, sys, nltk, operator, string, csv, os, gensim
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.collocations import *


DATA_DIR = "./wordnet"

# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* (<N>|<IN>)+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)* }
            """

LOC_PP = set(["in", "on", "at", "In", "On", "At"])
NP_NP = set(["the", "a", "that", "it", "to", "At", "The", "That", "It", "To"])
WHY_WHY = set(["because", "for", "to", "so", "Because", "For", "To", "So"])
WHO_N = set(["the", "The", "I", "a", "A"])
HOW_HOW = set([""])

VERBS = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
NEGATIONS = {"not", "n't"}

stemmer = SnowballStemmer("english")

# w2v = os.path.join("GoogleNews-vectors-negative300.bin")

#########################################################################################
# Chunking helper functions
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences

def pp_filter(subtree):
    return subtree.label() == "PP"

def np_filter(subtree):
    return subtree.label() == "NP"

def why_filter(subtree):
    return subtree.label() == "VP"

def who_filter(subtree):
    return subtree.label() == "NP"

def how_filter(subtree):
    return subtree.label() == "JJ"

def is_location(prep):
    return prep[0] in LOC_PP

def is_np(dt):
    return dt[0] in NP_NP

def is_why(y):
    return y[0] in WHY_WHY

def is_who(h):
    return h[0] in WHO_N

def is_how(h):
    return h[0] in HOW_HOW


def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations

    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)

    return locations

def find_np(tree):
    np = []
    for subtree in tree.subtrees(filter=np_filter):
        if is_np(subtree[0]):
            np.append(subtree)
    return np

def find_why(tree):
    why_p = []
    # print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    # print(tree)
    # print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    for subtree in tree.subtrees(filter=why_filter):
        # print("==================================")
        # print(subtree)
        # print("==================================")
        if is_why(subtree[0]):
            why_p.append(subtree)
    return why_p

def find_who(tree):
    who_p = []
    for subtree in tree.subtrees(filter=who_filter):
        if is_who(subtree[0]):
            who_p.append(subtree)
    return who_p

def find_candidates(target_sentences, chunker, text):
    candidates = []
    tokenized_question = nltk.word_tokenize(text.lower())
    for sent in target_sentences:
        tree = chunker.parse(sent)
        # print(tree)
        if tokenized_question[0] == "what":
            qType = "np_type"
            np = find_np(tree)
            candidates.extend(np)
        elif tokenized_question[0] == "where" or tokenized_question[0] == "when":
            qType = "loc_type"
            locations = find_locations(tree)
            candidates.extend(locations)
        elif tokenized_question[0] == "why":
            qType = "why_type"
            why = find_why(tree)
            candidates.extend(why)
        elif tokenized_question[0] == "who":
            qType = "who_type"
            who = find_who(tree)
            candidates.extend(who)

    return candidates


def find_sentences(patterns, sentences):
    # Get the raw text of each sentence to make it easier to search using regexes
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]

    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
        if matches:
            result.append(sent)

    return result

#########################################################################################

#########################################################################################
# Constituency helper functions
# See if our pattern matches the current root of the tree
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None
        return root

    return None

def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None

#########################################################################################

#########################################################################################
# Baseline helper functions

def get_bow(tagged_tokens, stopwords):
    return set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])

def get_ordered_bow(tagged_tokens, stopwords):
    return [stemmer.stem(t[0].lower()) for t in tagged_tokens if t[0].lower() not in stopwords]

def baseline(qbow, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]
    return best_answer

########################################################3
# Wordnet helper functions

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

########################################################################################
# My own written functions
def get_question_type(text):
    tokenized_question = nltk.word_tokenize(text.lower())
    if tokenized_question[0] == "what":
        return "np_type"
    if tokenized_question[0] == "where":
        return "loc_type"
    if tokenized_question[0] == "who":
        return "name_type"
    if tokenized_question[0] == "why":
        return "why_type"

def best_overlap_index(stemmed_ordered_qbow, stemmed_qbow, sentences, stopwords, question):
    answers = []
    for i in range (0, len(sentences)):
        sbow = get_bow(sentences[i], stopwords)
        ordered_sbow = get_ordered_bow(sentences[i], stopwords)

        # lemmas
        synset = set([])
        for word in sbow:
            word_synsets = wn.synsets(word)
            for word_synset in word_synsets:
                synset.add(word_synset.name()[0:word_synset.name().index(".")])
        sbow = sbow.union(synset)

        ordered_sbow_trigrams = list(nltk.trigrams(ordered_sbow))
        ordered_qbow_trigrams = list(nltk.trigrams(stemmed_ordered_qbow))
        # bigrams_overlap = len(set(ordered_sbow_bigrams) & set(ordered_qbow_bigrams))

        trigrams_overlap = 0
        for x in ordered_qbow_trigrams:
            x0_synsets = extract_synset(wn.synsets(x[0]))
            x0_synsets.add(x[0])
            x1_synsets = extract_synset(wn.synsets(x[1]))
            x1_synsets.add(x[1])
            x2_synsets = extract_synset(wn.synsets(x[2]))
            x2_synsets.add(x[2])

            for y in ordered_sbow_trigrams:
                y0_synsets = extract_synset(wn.synsets(y[0]))
                y0_synsets.add(y[0])
                y1_synsets = extract_synset(wn.synsets(y[1]))
                y1_synsets.add(y[1])
                y2_synsets = extract_synset(wn.synsets(y[2]))
                y2_synsets.add(y[2])
                if len(x0_synsets & y0_synsets) > 0 and len(x1_synsets & y1_synsets) > 0 and len(x2_synsets & y2_synsets) > 0:
                    trigrams_overlap += 1

        overlap = len(stemmed_qbow & sbow)
        answers.append((overlap+trigrams_overlap, i, sentences[i]))

        # if i == 11 and question["qid"] == 'mc500.train.18.18':
        #     print(ordered_sbow_trigrams)
        #     print(ordered_qbow_trigrams)
        #     print(trigrams_overlap)
        #     print(overlap+trigrams_overlap)

    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    # return best index for the most overlap of WORDS
    best_answer = (answers[0])[1]
    # ###########################################################
    # if question["qid"] == 'mc500.train.18.18':
    #     print(len(answers))
    #     print("")
    #     for it in answers:
    #          print(it)
    # ###########################################################
    return best_answer

def get_pattern_bow(tagged_tokens, stopwords):
    # for t in tagged_tokens:
    #     if t[0].lower() not in stopwords:
    #         print(t[1])
    return [t[1] for t in tagged_tokens if t[0].lower() not in stopwords]

def extract_synset(synsets):
    l = set([])
    for synset in synsets:
        hypo = synset.hyponyms()
        for item in hypo:
            l.add(item.name()[0:item.name().index(".")])
        hyper = synset.hypernyms()
        for item in hyper:
            l.add(item.name()[0:item.name().index(".")])
        lemm = synset.lemmas()
        for item in lemm:
            l.add(item.name()[0:synset.name().index(".")])
    return l

########################################################################################
def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###
    # Our tools

    # stemmer = SnowballStemmer("english")
    chunker = nltk.RegexpParser(GRAMMAR)

    driver = QABase()

    # question["qid"] returns the form: "fables-04-7"
    q = driver.get_question(question["qid"])
    current_story = driver.get_story(q["sid"])

    #############################################
    # if question["qid"] == 'blogs-03-1':
    #     print(question["text"])
    #     print(sent_tokenized_text[0])
    #     print("++++++++++++++++++++++++++++++++++++++++++++++")
    ############################################

    stopwords = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))

    if question["difficulty"] == 'Medium' or question["difficulty"] == 'Easy':

        if question["type"] != 'Story':
            sentences = get_sentences(current_story["sch"])
        else:
            sentences = get_sentences(current_story["text"])

        Q = nltk.word_tokenize(question["text"].lower())
        # print(Q)

        all_stemmed_sentences = []
        for sent in sentences:
            temp_sent = []
            for w, pos in sent:
                temp_sent.append((stemmer.stem(w), pos))
            all_stemmed_sentences.append(temp_sent)

        # prepare qbow for word-overlapping
        qbow = get_bow(get_sentences(question["text"])[0], stopwords)
        stemmed_qbow = []
        for w in qbow:
            stemmed_qbow.append(stemmer.stem(w))
        stemmed_qbow = set(stemmed_qbow)
        # print(stemmed_qbow)

        # make ordered qbow for bigram and trigram matching
        stemmed_ordered_qbow = get_ordered_bow(get_sentences(question["text"])[0], stopwords)

        # prepare pattern_qbow for pattern overlapping
        # pattern_qbow = get_pattern_bow(get_sentences(question["text"])[0], stopwords)

        # if question["qid"] == 'mc500.train.18.18':
        #     print("stemmed_qbow:", stemmed_qbow)
        #     print("pattern_qbow:", pattern_qbow)

        best_idx = best_overlap_index(stemmed_ordered_qbow, stemmed_qbow, all_stemmed_sentences, stopwords, question)
        # print(question["qid"], best_idx)

        if question["type"] != 'Story':
            tree = current_story["sch_par"][best_idx]
        else:
            tree = current_story["story_par"][best_idx]

        #############################################
        # if question["qid"] == 'blogs-03-13':
        #     print(Q)
        #     print(tree)
        #     print("++++++++++++++++++++++++++++++++++++++++++++++")
        ############################################
        # print(tree)
        # Create our pattern

        #########################################
        # MAKE PATTERN FIT FOR TYPE OF QUESTION #
        #########################################
        # print(Q[0])
        if ('where' in Q) or ('when' in Q) :
            pattern = nltk.ParentedTree.fromstring("(PP)")
        elif 'who' in Q or ('which' in Q):
            pattern = nltk.ParentedTree.fromstring("(NP (DT) (*) (NN))")
        elif ('what' in Q):
            pattern = nltk.ParentedTree.fromstring("(VP (*) (NP))")
        elif 'why' in Q:
            pattern = nltk.ParentedTree.fromstring("(SBAR)")
        elif 'how' in Q:
            pattern = nltk.ParentedTree.fromstring("(RB)")
            # don't know how to deal with 'did' questions
        elif 'did' in Q:
            pattern = nltk.ParentedTree.fromstring("(ROOT)")
        else:
            return doBaseline(question, story)

        subtree1 = pattern_matcher(pattern, tree)

        ############################################
        # if question["qid"] == 'mc500.train.25.3':
        #     print(Q)
        #     print(tree)
        #     print("subtree1")
        #     print(subtree1)
        ############################################
        if subtree1 == None:
            #######################################
            answer = doBaseline(question, story)
            # answer = "doBaseline"
            #######################################
        else:
            if ('where' in Q) or ('when' in Q):
                pattern = nltk.ParentedTree.fromstring("(PP)")
            elif 'who' in Q or ('which' in Q):
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif 'what' in Q:
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif 'why' in Q:
                pattern = nltk.ParentedTree.fromstring("(SBAR)")
            elif 'how' in Q:
                pattern = nltk.ParentedTree.fromstring("(RB)")

                # don't know how to deal with 'did' questions
            elif 'did' in Q:
                pattern = nltk.ParentedTree.fromstring("(ROOT)")


            # Find and make the answer
            # print(subtree)
            subtree2 = pattern_matcher(pattern, subtree1)
            if subtree2 == None:
                #######################################
                answer = doBaseline(question, story)
                # answer = "doBaseline"
                #######################################
            else:
                answer = " ".join(subtree2.leaves())

            ############################################
            # if question["qid"] == 'mc500.train.18.18':
            #     print("subtree2")
            #     print(subtree2)
            ############################################
            # cheat for dealing with 'did' questions
            if Q[0] == 'did':
                negations = len(set(nltk.word_tokenize(answer)) & NEGATIONS)
                if negations > 0:
                    answer = "no"
                else:
                    answer = "yes"


    elif question["difficulty"] == 'Hard' or question["difficulty"] == 'Discourse':

        if question["type"] != 'Story':
            sentences = get_sentences(current_story["sch"])
        else:
            sentences = get_sentences(current_story["text"])

        Q = nltk.word_tokenize(question["text"].lower())
        # print(Q)

        all_stemmed_sentences = []
        for sent in sentences:
            temp_sent = []
            for w, pos in sent:
                temp_sent.append((stemmer.stem(w), pos))
            all_stemmed_sentences.append(temp_sent)

        qbow = get_bow(get_sentences(question["text"])[0], stopwords)
        ordered_qbow = get_ordered_bow(get_sentences(question["text"])[0], stopwords)
        stemmed_qbow = []
        for w in qbow:
            stemmed_qbow.append(stemmer.stem(w))
        stemmed_qbow = set(stemmed_qbow)


        stemmed_ordered_qbow = get_ordered_bow(get_sentences(question["text"])[0], stopwords)

        joined_grams = []
        # create bigrams and trigrams, then find collocations
        if len(stemmed_qbow) >= 2:
            bigrams = list(nltk.bigrams(stemmed_ordered_qbow))
            joined_grams += ['_'.join(b) for b in bigrams]
        if len(stemmed_qbow) > 2:
            trigrams = list(nltk.trigrams(stemmed_ordered_qbow))
            joined_grams += ['_'.join(t) for t in trigrams]

        stemmed_qbow = stemmed_qbow.union(set(joined_grams))


        #######################################
        # Collect hypernyms, hyponyms, lemmas #
        #######################################
        noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
        verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

        # {synset_id : {synset_offset: X, noun/verb: Y, stories: set(Z)}}, ...}
        # e.g. {help.v.01: {synset_offset: 2547586, noun: aid, stories: set(Z)}}, ...
        # noun_ids = pickle.load(open("Wordnet_nouns.dict", "rb"))
        # verb_ids = pickle.load(open("Wordnet_verbs.dict", "rb"))

        ####################################################################################
        # My own code documentations:
        # items is a dictionary, per synset_id, we have:
        #                   {'synset_offset': '7-digit number',
        #                   'story_noun': 'each noun word correlated with synset_id',
        #                   'stories': "'story-id.vgl'"}
        ####################################################################################

        # iterate through dictionary
        for synset_id, items in noun_ids.items():
            noun = items['story_noun']
            stories = items['stories']
            # print(noun, stories)
            # get lemmas, hyponyms, hypernyms

        for synset_id, items in verb_ids.items():
            verb = items['story_verb']
            stories = items['stories']
            # print(verb, stories)
            # get lemmas, hyponyms, hypernyms

        hypo_dict = {}
        hyper_dict = {}
        lemma_dict = {}

        for word in stemmed_qbow:
            word_synsets = wn.synsets(word)

            # hyponyms
            temp_this_word_hyponyms = []
            for word_synset in word_synsets:
                word_hypo = word_synset.hyponyms()
                temp_curr_hyponyms = []
                for hypo in word_hypo:
                    temp_curr_hyponyms.append(hypo.name()[0:hypo.name().index(".")])
                temp_this_word_hyponyms += temp_curr_hyponyms
            hypo_dict[word] = temp_this_word_hyponyms

            # hyperyms
            temp_this_word_hypernyms = []
            for word_synset in word_synsets:
                word_hyper = word_synset.hypernyms()
                temp_curr_hypernyms = []
                for hyper in word_hyper:
                    temp_curr_hypernyms.append(hyper.name()[0:hyper.name().index(".")])
                temp_this_word_hypernyms += temp_curr_hypernyms
            hyper_dict[word] = temp_this_word_hypernyms

            # lemmas
            temp_this_word_lemmas = [word]
            for word_synset in word_synsets:
                temp_this_word_lemmas.append(word_synset.name()[0:word_synset.name().index(".")])
            lemma_dict[word] = temp_this_word_lemmas

        # combine hyponyms, hypernyms, lemmas with stemmed_qbow
        
        # hyponyms
        syn_list = set([])
        for stemmed_qbow_word in stemmed_qbow:
            for hypo in hypo_dict[stemmed_qbow_word]:
                syn_list.add(hypo)

        # hypernyms
        for stemmed_qbow_word in stemmed_qbow:
            for hyper in hyper_dict[stemmed_qbow_word]:
                syn_list.add(hyper)
                # if question["qid"] == 'fables-06-14':
                #     print(stemmed_qbow_word)
                #     print(hyper)

        # lemmas
        for stemmed_qbow_word in stemmed_qbow:
            for lemma in lemma_dict[stemmed_qbow_word]:
                syn_list.add(lemma)

        stemmed_qbow = stemmed_qbow.union(syn_list)

        best_idx = best_overlap_index(stemmed_ordered_qbow, stemmed_qbow, all_stemmed_sentences, stopwords, question)
        # print(question["qid"], best_idx)

        if question["type"] != 'Story':
            tree = current_story["sch_par"][best_idx]
        else:
            tree = current_story["story_par"][best_idx]

        #############################################
        # if question["qid"] == 'blogs-03-13':
        #     print(Q)
        #     print(tree)
        #     print("++++++++++++++++++++++++++++++++++++++++++++++")
        ############################################
        # print(tree)
        # Create our pattern

        #########################################
        # MAKE PATTERN FIT FOR TYPE OF QUESTION #
        #########################################
        # print(Q[0])
        if ('where' in Q) or ('when' in Q):
            pattern = nltk.ParentedTree.fromstring("(PP (*) (NP))")
        elif 'who' in Q or ('which' in Q):
            pattern = nltk.ParentedTree.fromstring("(NP (DT) (*) (NN))")
        elif 'what' in Q:
            pattern = nltk.ParentedTree.fromstring("(NP)")
        elif 'why' in Q:
            pattern = nltk.ParentedTree.fromstring("(SBAR)")
        elif 'how' in Q:
            pattern = nltk.ParentedTree.fromstring("(RB)")
            # don't know how to deal with 'did' questions
        elif 'did' in Q:
            pattern = nltk.ParentedTree.fromstring("(ROOT)")
        else:
            return doBaseline(question, story)

        subtree1 = pattern_matcher(pattern, tree)

        #################################################
        # who_qs = ["fables-03-22", "fables-03-23", "fables-03-25", "fables-03-26", "mc500.train.25.3"]
        where_qs = ["blogs-03-15", "blogs-03-19", "blogs-05-18", "fables-03-27", "mc500.train.0.23", "mc500.train.0.24", "mc500.train.18.23", "mc500.train.18.25", "mc500.train.111.5"]
        if question["qid"] in where_qs:
            print(Q)
            print(tree)
            print("subtree1")
            print(subtree1)
        ######################################################################
        if subtree1 == None:
            #######################################
            answer = doBaseline(question, story)
            # answer = "doBaseline"
            #######################################
        else:
            # create a new pattern to match a smaller subset of subtrees
            if ('where' in Q) or ('when' in Q):
                pattern = nltk.ParentedTree.fromstring("(PP)")
            elif 'who' in Q or ('which' in Q):
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif 'what' in Q:
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif 'why' in Q:
                pattern = nltk.ParentedTree.fromstring("(SBAR)")
            elif 'how' in Q:
                pattern = nltk.ParentedTree.fromstring("(RB)")

                # don't know how to deal with 'did' questions
            elif 'did' in Q:
                pattern = nltk.ParentedTree.fromstring("(ROOT)")


            # Find and make the answer
            # print(subtree)
            subtree2 = pattern_matcher(pattern, subtree1)

            ####################################
            if question["qid"] in where_qs:
                print(pattern)
                print("subtree2")
                print(subtree2)
            ###################################

            if subtree2 == None:
                #######################################
                answer = doBaseline(question, story)
                # answer = "doBaseline"
                #######################################
            else:
                answer = " ".join(subtree2.leaves())

            ############################################
            # if question["qid"] == 'mc500.train.18.18':
            #     print("subtree2")
            #     print(subtree2)
            ############################################
            # cheat for dealing with 'did' questions
            if Q[0] == 'did':
                negations = len(set(nltk.word_tokenize(answer)) & NEGATIONS)
                if negations > 0:
                    answer = "no"
                else:
                    answer = "yes"

    else:
        #########################################
        answer = doBaseline(question, story)
        # answer = "doBaseline"
        #########################################

    ###     End of Your Code         ###

    return answer

def doBaseline(question, story):
    text = story["text"]
    questions = question["text"]
    stopwords = set(nltk.corpus.stopwords.words("english"))
    qbow = get_bow(get_sentences(questions)[0], stopwords)
    # get_bow = filters stopwords, returns
    # get_sentences returns tagged question, in this case, only the first question
    # qbow therefore is a a list of tagged words from the question without stopwords

    sentences = get_sentences(text)
    answer_tuples = baseline(qbow, sentences, stopwords)
    answer = " ".join(t[0] for t in answer_tuples)

    return answer


#############################################################
###     Dont change the code below here
#############################################################

class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run() #reads questions, iterates over questions
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
