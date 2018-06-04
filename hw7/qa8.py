import re, sys, nltk, operator
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tree import Tree

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
NP_NP = set(["a", "the", "that", "it", "to", "At", "The", "That", "It", "To"])
WHY_WHY = set(["because", "for", "to", "so", "Because", "For", "To", "So"])
WHO_N = set(["the", "The", "I", "a", "A"])
HOW_HOW = set([""])

VERBS = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

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

########################################################################################

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

def best_overlap_index(qbow, sentences, stopwords, question):
    answers = []
    for i in range (0, len(sentences)):
        sbow = get_bow(sentences[i], stopwords)
        overlap = len(qbow & sbow)
        answers.append((overlap, i, sentences[i]))

    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    ###########################################################
    # if question["qid"] == 'fables-04-6':
    #     print("answers:", answers)
    #     print(len(answers))
    ###########################################################
    # return best index for the most overlap
    best_answer = (answers[0])[1]
    return best_answer

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

    stemmer = SnowballStemmer("english")
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()

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

    stopwords = set(nltk.corpus.stopwords.words("english"))


    if (question["difficulty"] == 'Easy'):



        if question["type"] != 'Story':
            sentences = get_sentences(current_story["sch"])
            text = story["sch"]
            text = nltk.sent_tokenize(text)

        else:
            sentences = get_sentences(current_story["text"])
            text = story["text"]
            text = nltk.sent_tokenize(text)

        Q = nltk.word_tokenize(question["text"].lower())
        # print(Q)

        all_stemmed_sentences = []
        for sent in sentences:
            temp_sent = []
            for w, pos in sent:
                temp_sent.append((stemmer.stem(w), pos))
            all_stemmed_sentences.append(temp_sent)
        stop_words = set(nltk.corpus.stopwords.words("english"))
        qbow = get_bow(get_sentences(question["text"])[0], stopwords)
        stemmed_qbow = []
        for w in qbow:
            stemmed_qbow.append(stemmer.stem(w))
        stemmed_qbow = set(stemmed_qbow)
        best_idx = best_overlap_index(stemmed_qbow, all_stemmed_sentences, stop_words, question)
        # print(question["qid"], best_idx)

        # tokenize questions, also removing punctuations to extract keywords
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized_question_text = tokenizer.tokenize(question["text"])
        tagged_tokenized_question_text = nltk.pos_tag(tokenized_question_text)

        # remove stopwords
        tagged_keywords_list = []

        for word, tag in tagged_tokenized_question_text:
            if word not in stopwords:
                tagged_keywords_list.append((word, tag))

        # lemmatize keywords
        lemmatized_keywords_list = []
        for keyword, tag in tagged_keywords_list:
            lemmatized_keywords_list.append(stemmer.stem(keyword))

        #####################################################
        # if question["qid"] == 'fables-04-6':
        #     print("text:", text)
        #     print("best index:", best_idx)
        #     print("qid:", question["qid"])
        #     print(text[best_idx])
        #     print("==============================")
        #     print(get_sentences("".join(text)))
        #####################################################


        best_sent = get_sentences(text[best_idx])

        # Find the sentences that have all of our keywords in them
        # Last time, 2nd arg is sentences = get_sentences(text) which returns tuple of each word
        target_sentences = find_sentences(lemmatized_keywords_list, best_sent)
        # Extract the candidate locations from these sentences
        candidates_forest = find_candidates(target_sentences, chunker, question["text"])

        if len(candidates_forest) == 0:
            answer = doBaseline(question, story)
        else:

            possible_answers_list = []

            # locations is a list of trees
            for candidate in candidates_forest:
                # candidate.draw()
                possible_answers_list.append(" ".join([token[0] for token in candidate.leaves()]))
            answer = " ".join(possible_answers_list)

            ###########################################
            # currently, possible_answer contains the actual needed answer,
            # plus some garbage words around it from chunking,
            # we might be able to filter this out SOMEHOW
            # possible_answer is a list of strings
            ###########################################


    elif question["difficulty"] == 'Medium':

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
        stop_words = set(nltk.corpus.stopwords.words("english"))
        qbow = get_bow(get_sentences(question["text"])[0], stopwords)
        stemmed_qbow = []
        for w in qbow:
            stemmed_qbow.append(stemmer.stem(w))
        stemmed_qbow = set(stemmed_qbow)
        best_idx = best_overlap_index(stemmed_qbow, all_stemmed_sentences, stop_words, question)
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
        if Q[0] == 'where' or Q[0] == 'when':
            pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")
        elif Q[0] == 'who':
            pattern = nltk.ParentedTree.fromstring("(NP)")
        elif Q[0] == 'what':
            pattern = nltk.ParentedTree.fromstring("(NP)")
        elif Q[0] == 'why':
            pattern = nltk.ParentedTree.fromstring("(SBAR)")
        elif Q[0] == 'how':
            pattern = nltk.ParentedTree.fromstring("(RB)")

        # don't know how to deal with 'did' questions
        elif Q[0] == 'did':
            pattern = nltk.ParentedTree.fromstring("(S)")

        subtree1 = pattern_matcher(pattern, tree)

        ############################################
        # if question["qid"] == 'blogs-03-13':
        #     print("subtree1")
        #     print(subtree1)
        ############################################
        if subtree1 == None:
            #######################################
            answer = doBaseline(question, story)
            # answer = "doBaseline"
            #######################################
        else:
            # create a new pattern to match a smaller subset of subtrees
            if Q[0] == 'where' or Q[0] == 'when':
                pattern = nltk.ParentedTree.fromstring("(VP)")
            elif Q[0] == 'who':
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif Q[0] == 'what':
                pattern = nltk.ParentedTree.fromstring("(NP)")
            elif Q[0] == 'why':
                pattern = nltk.ParentedTree.fromstring("(SBAR (IN) (S))")
            elif Q[0] == 'how':
                pattern = nltk.ParentedTree.fromstring("(RB)")

            # don't know how to deal with 'did' questions
            elif Q[0] == 'did':
                pattern = nltk.ParentedTree.fromstring("(S)")


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
                answer = "yes"

    elif question["difficulty"] == 'Hard':

        answer = "h"


    elif question["difficulty"] == 'Discourse':

        answer = "h"


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
