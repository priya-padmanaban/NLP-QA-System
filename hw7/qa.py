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
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """
NOUNS = ['NN', 'NNP', 'NNS']
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


LOC_PP = set(["in", "on", "at"])
NP_NP = set(["a", "the", "that", "it"])
WHY_WHY = set(["because"])

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

def is_location(prep):
    return prep[0] in LOC_PP

def is_np(prep):
    return prep[0] in NP_NP


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


def find_candidates(crow_sentences, chunker, text):
    candidates = []
    tokenized_question = nltk.word_tokenize(text.lower())

    for sent in crow_sentences:
        tree = chunker.parse(sent)
        # print(tree)
        if tokenized_question[0] == "what":
            qType = "np_type"
            np = find_np(tree)
            candidates.extend(np)
        elif tokenized_question[0] == "where":
            qType = "loc_type"
            locations = find_locations(tree)
            candidates.extend(locations)
        else:
            locations = find_locations(tree)
            candidates.extend(locations)
        '''
        if tokenized_question[0] == "who":
            qType = "name_type"
            name = find_np(tree)
            candidates.extend(name)
        if tokenized_question[0] == "why":
            qType = "why_type"
            why = find_np(tree)
            candidates.extend(why)
        '''
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
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


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
    q = driver.get_question(question["qid"])
    current_story = driver.get_story(q["sid"])
    text = story["text"]

    # Apply the standard NLP pipeline we've seen before
    sentences = get_sentences(text)
    # print(sentences)
    # print(question["text"])

    # tokenize questions, also removing punctuations to extract keywords
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_question_text = tokenizer.tokenize(question["text"])
    tagged_tokenized_question_text = nltk.pos_tag(tokenized_question_text)
    # remove stopwords
    tagged_keywords_list = []
    stopwords = set(nltk.corpus.stopwords.words("english"))
    for word, tag in tagged_tokenized_question_text:
        if word not in stopwords:
            tagged_keywords_list.append((word, tag))

    # lemmatize keywords
    ######################### KEYWORDS MUST BE IN A SPECIFIC ORDER, THIS IS RANDOM
    ######################### TAGGING FOR SINGLE WORDS ARE USUALLY TREATED AS NOUNS EVEN IF THEY SHOULD BE VERBS
    lemmatized_keywords_list = []

    for keyword, tag in tagged_keywords_list:
        lemmatized_keywords_list.append(stemmer.stem(keyword))

    # sort into noun, verb order

    crow_sentences = find_sentences(lemmatized_keywords_list, sentences)
    # crow_sentences = find_sentences(keywords_list, sentences)
    # print(crow_sentences)
    # Extract the candidate locations from these sentences
    locations = find_candidates(crow_sentences, chunker, question["text"])
    # print("sentences:", len(sentences))
    # print("orignal keywords:", tagged_keywords_list)
    # print("keywords:", lemmatized_keywords_list)
    #
    # print("crow_sentences:", len(crow_sentences))
    # print(question["text"], locations)

    if question["difficulty"] == 'Easy' and len(locations) != 0:
        '''
        if story["sid"] == "fables-01":
            print("-----------------------------------------------------")
            print(crow_sentences)
            print("keywords:", keywords_list)
            print("questions:", question["text"])
            print("loc:", locations)
            # Print them out
            for loc in locations:
                print(loc)
                print(" ".join([token[0] for token in loc.leaves()]))
            print("-----------------------------------------------------")
        '''
        answer = []

        for loc in locations:
            answer.append(" ".join([token[0] for token in loc.leaves()]))
        answer = " ".join(answer)

    elif question["difficulty"] == 'Medium':
        sid_content = driver.get_story(story["sid"])

        if len(sid_content["sch_par"]) != 0:
            tree = sid_content["sch_par"][1]
            # print(tree)
            # Create our pattern
            pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")

            # # Match our pattern to the tree
            subtree = pattern_matcher(pattern, tree)

        # print(subtree)
        # print(" ".join(subtree.leaves()))
        if len(sid_content["sch_par"]) == 0 or subtree == None:
            answer = doBaseline(question, story)
        else:
            # create a new pattern to match a smaller subset of subtree
            pattern = nltk.ParentedTree.fromstring("(PP)")

            # Find and print the answer
            subtree2 = pattern_matcher(pattern, subtree)
            answer = " ".join(subtree2.leaves())

    else:
        answer = doBaseline(question, story)
        

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
    # print("question:", questions)
    # print(answer)


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
