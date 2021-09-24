from sklearn.preprocessing.data import binarize
from oie_readers.argument import Argument
from operator import itemgetter
from collections import defaultdict
import nltk
import itertools
import logging
import numpy as np
import pdb

class Extraction:
    def __init__(self, pred, head_pred_index, sent, confidence, question_dist = '', index = -1):
        self.pred = pred
        self.head_pred_index = head_pred_index
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        self.indsForQuestions = defaultdict(lambda: set())
        self.is_mwp = False
        self.question_dist = question_dist
        self.index = index

    def distArgFromPred(self, arg):
        assert(len(self.pred) == 2)
        dists = []
        for x in self.pred[1]:
            for y in arg.indices:
                dists.append(abs(x - y))

        return min(dists)

    def argsByDistFromPred(self, question):
        return sorted(self.questions[question], key = lambda arg: self.distArgFromPred(arg))

    def addArg(self, arg, question = None):
        self.args.append(arg)
        if question:
            self.questions[question] = self.questions.get(question,[]) + [Argument(arg)]

    def noPronounArgs(self):
        for (a, _) in self.args:
            tokenized_arg = nltk.word_tokenize(a)
            if len(tokenized_arg) == 1:
                _, pos_tag = nltk.pos_tag(tokenized_arg)[0]
                if ('PRP' in pos_tag):
                    return False
        return True

    def isContiguous(self):
        return all([indices for (_, indices) in self.args])

    def toBinary(self):
        ''' Try to represent this extraction's arguments as binary
        If fails, this function will return an empty list.  '''

        ret = [self.elementToStr(self.pred)]

        if len(self.args) == 2:
            return ret + [self.elementToStr(arg) for arg in self.args]

        return []

        if not self.isContiguous():
            return []

        binarized = self.binarizeByIndex()

        if binarized:
            return ret + binarized

        return []


    def elementToStr(self, elem, print_indices = True):
        ''' formats an extraction element (pred or arg) as a raw string
        removes indices and trailing spaces '''
        if print_indices:
            return str(elem)
        if isinstance(elem, str):
            return elem
        if isinstance(elem, tuple):
            ret = elem[0].rstrip().lstrip()
        else:
            ret = ' '.join(elem.words)
        assert ret, "empty element? {0}".format(elem)
        return ret

    def binarizeByIndex(self):
        extraction = [self.pred] + self.args
        markPred = [(w, ind, i == 0) for i, (w, ind) in enumerate(extraction)]
        sortedExtraction = sorted(markPred, key = lambda ws, indices, f : indices[0])
        s =  ' '.join(['{1} {0} {1}'.format(self.elementToStr(elem), SEP) if elem[2] else self.elementToStr(elem) for elem in sortedExtraction])
        binArgs = [a for a in s.split(SEP) if a.rstrip().lstrip()]

        if len(binArgs) == 2:
            return binArgs

        return []

    def bow(self):
        return ' '.join([self.elementToStr(elem) for elem in [self.pred] + self.args])

    def getSortedArgs(self):
        if self.question_dist:
            return self.sort_args_by_distribution()
        ls = []
        for q, args in self.questions.iteritems():
            if (len(args) != 1):
                logging.debug("Not one argument: {}".format(args))
                continue
            arg = args[0]
            indices = list(self.indsForQuestions[q].union(arg.indices))
            if not indices:
                logging.debug("Empty indexes for arg {} -- backing to zero".format(arg))
                indices = [0]
            ls.append(((arg, q), indices))
        return [a for a, _ in sorted(ls,
                                     key = lambda indices: min(indices[1]))]

    def question_prob_for_loc(self, question, loc):
        gen_question = generalize_question(question)
        q_dist = self.question_dist[gen_question]
        logging.debug("distribution of {}: {}".format(gen_question,
                                                      q_dist))

        return float(q_dist.get(loc, 0)) /  \
            sum(q_dist.values())

    def sort_args_by_distribution(self):
        INF_LOC = 100 

        ret = {INF_LOC: []}
        logging.debug("sorting: {}".format(self.questions))

        logging.debug("probs for subject: {}".format([(q, self.question_prob_for_loc(q, 0))
                                                      for (q, _) in self.questions.iteritems()]))

        subj_question, subj_args = max(self.questions.iteritems(),
                                       key = lambda q: self.question_prob_for_loc(q[0], 0))

        ret[0] = [(subj_args[0], subj_question)]

        for (question, args) in sorted([(q, a)
                                        for (q, a) in self.questions.iteritems() if (q not in [subj_question])],
                                       key = lambda q: \
                                       sum(self.question_dist[generalize_question(q[0])].values()),
                                       reverse = True):
            gen_question = generalize_question(question)
            arg = args[0]
            assigned_flag = False
            for (loc, count) in sorted(self.question_dist[gen_question].iteritems(),
                                       key = lambda c: c[1],
                                       reverse = True):
                if loc not in ret:
                    ret[loc] = [(arg, question)]
                    assigned_flag = True
                    break

            if not assigned_flag:
                logging.debug("Couldn't find an open assignment for {}".format((arg, gen_question)))
                ret[INF_LOC].append((arg, question))

        logging.debug("Linearizing arg list: {}".format(ret))

        return [arg
                for (_, arg_ls) in sorted(ret.iteritems(),
                                          key = lambda k, v: int(k))
                for arg in arg_ls]


    def __str__(self):
        pred_str = self.elementToStr(self.pred)
        return '{}\t{}\t{}'.format(self.get_base_verb(pred_str),
                                   self.compute_global_pred(pred_str,
                                                            self.questions.keys()),
                                   '\t'.join([escape_special_chars(self.augment_arg_with_question(self.elementToStr(arg),
                                                                                                  question))
                                              for arg, question in self.getSortedArgs()]))

    def get_base_verb(self, surface_pred):
        return surface_pred.split(' ')[-1]


    def compute_global_pred(self, surface_pred, questions):
        from operator import itemgetter
        split_surface = surface_pred.split(' ')

        if len(split_surface) > 1:
            verb = split_surface[-1]
            ret = split_surface[:-1] 
        else:
            verb = split_surface[0]
            ret = []

        split_questions = map(lambda question: question.split(' '),
                            questions)

        preds = map(normalize_element,
                    map(itemgetter(QUESTION_TRG_INDEX),
                        split_questions))
        if len(set(preds)) > 1:
            ret.append(verb)

        if len(set(preds)) == 1:
            ret.append(preds[0])

            pps = map(normalize_element,
                      map(itemgetter(QUESTION_PP_INDEX),
                          split_questions))

            obj2s = map(normalize_element,
                        map(itemgetter(QUESTION_OBJ2_INDEX),
                            split_questions))

            if (len(set(pps)) == 1):
                self.is_mwp = True 
                ret.append(pps[0])

        return " ".join(ret).strip()


    def augment_arg_with_question(self, arg, question):
        wh, aux, sbj, trg, obj1, pp, obj2 = map(normalize_element,
                                                question.split(' ')[:-1]) 

        if (not self.is_mwp) and pp and (not obj2):
            if not(arg.startswith("{} ".format(pp))):
                return " ".join([pp,
                                 arg])

        return arg

    def clusterScore(self, cluster):
        logging.debug("*-*-*- Cluster: {}".format(cluster))

        arr = np.array([x for ls in cluster for x in ls])
        centroid = np.sum(arr)/arr.shape[0]
        logging.debug("Centroid: {}".format(centroid))

        return np.average([max([abs(x - centroid) for x in ls]) for ls in cluster])

    def resolveAmbiguity(self):

        elements = [self.pred] \
                   + [(s, indices)
                      for (s, indices)
                      in self.args
                      if indices]
        logging.debug("Resolving ambiguity in: {}".format(elements))

        all_combinations = list(itertools.product(*map(itemgetter(1), elements)))
        logging.debug("Number of combinations: {}".format(len(all_combinations)))

        resolved_elements = zip(map(itemgetter(0), elements),
                                min(all_combinations,
                                    key = lambda cluster: self.clusterScore(cluster)))
        logging.debug("Resolved elements = {}".format(resolved_elements))

        self.pred = resolved_elements[0]
        self.args = resolved_elements[1:]

    def conll(self, external_feats = {}):
        return '\n'.join(["\t".join(map(str,
                                        [i, w] + \
                                        list(self.pred) + \
                                        [self.head_pred_index] + \
                                        external_feats + \
                                        [self.get_label(i)]))
                          for (i, w)
                          in enumerate(self.sent.split(" "))]) + '\n'

    def get_label(self, index):
        ent = [(elem_ind, elem)
               for (elem_ind, elem)
               in enumerate(map(itemgetter(1),
                                [self.pred] + self.args))
               if index in elem]

        if not ent:
            return "O"

        if len(ent) > 1:
            logging.warn("Index {} appears in one than more element: {}".\
                         format(index,
                                "\t".join(map(str,
                                              [ent,
                                               self.sent,
                                               self.pred,
                                               self.args]))))


        elem_ind, elem = min(ent, key = lambda ls: len(ls[1]))

        prefix = "P" if elem_ind == 0 else "A{}".format(elem_ind - 1)

        suffix = "B" if index == elem[0] else "I"

        return "{}-{}".format(prefix, suffix)

    def __str__(self):
        return '{0}\t{1}'.format(self.elementToStr(self.pred,
                                                   print_indices = True),
                                 '\t'.join([self.elementToStr(arg)
                                            for arg
                                            in self.args]))

flatten = lambda l: [item for sublist in l for item in sublist]


def normalize_element(elem):
    return elem.replace("_", " ") \
        if (elem != "_")\
           else ""

def escape_special_chars(s):
    return s.replace('\t', '\\t')


def generalize_question(question):
    import nltk   
    wh, aux, sbj, trg, obj1, pp, obj2 = question.split(' ')[:-1] 
    return ' '.join([wh, sbj, obj1])



SEP = ';;;'
QUESTION_TRG_INDEX =  3 
QUESTION_PP_INDEX = 5
QUESTION_OBJ2_INDEX = 6
