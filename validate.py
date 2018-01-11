#!/usr/bin/python
from __future__ import division
from collections import defaultdict, Counter
import numpy as np
import random, re, math, json, glob
import NER
import cPickle as pickle
from copy import deepcopy
from nltk.tag.perceptron import PerceptronTagger

import subprocess, os

tagger = PerceptronTagger()

def split_data(filename, n):
    """takes the data we have and splits it into n random subsets in order
    to perform cross-validation"""

    #add the training data
    with open(filename, "r") as f:
        #/data/2017/wnut17train.conll
        records = re.split("\n[\t]?\n", f.read().strip())
    if n == 1:  #just turn this into a test function, no splitting
        data_split = records
    else:
        random.seed(0) #keep this for testing
        random.shuffle(records)     #randomize the order of the data
        data_split = [[]
                      for i in range(n)]
        for i, record in enumerate(records):
            batch_num = i % n
            data_split[batch_num].append(record)

    return data_split

def crossTrain(n = 10, multi = True):
    """to maximize efficiency this will create _exdeflike_ and
    _indeflike_ once and write them to JSON files so we can just
    call them later without having to train the same thing over
    and over"""

    training_2016 = '../data/2016/data/train'
    dev_2016 = '../data/2016/data/dev'
    test_2016 = '../data/2016/data/test'
    training_2017 = '../data/2017/wnut17train.conll'

    data_split = split_data(training_2017, n)
    total_indef = defaultdict(defaultdict)
    total_exdef = defaultdict(defaultdict)

    #we'll use the 2016 data in addition to do the training
    # with open(training_2016, 'r') as f:
    #     data_2016 = re.split("\n[\t]?\n", f.read().strip())
    # with open(dev_2016, 'r') as f1:
    #     dev_2016 = re.split("\n[\t]?\n", f1.read().strip())
    # with open(test_2016, 'r') as f2:
    #     test_2016 = re.split("\n[\t]?\n", f2.read().strip())

    # #add these up
    # train2 = data_2016 + dev_2016 + test_2016
    """
    print "extra con"
    with open('extracon.txt', 'rb') as infile:
        con_counts = pickle.load(infile)
    print "extra def"
    with open('extradef.txt', 'rb') as f:
        deflike = pickle.load(f)
    """
    for i, data in enumerate(data_split):
        #need to perform testing n times giving each partition of the data
        #a chance to be the test data and the rest training
        #con_counts_copy = deepcopy(con_counts)
        #deflike_copy = deepcopy(deflike)

        copy_datasplit = list(data_split)
        test = copy_datasplit.pop(i)
        #now flatten the list of n-1 lists to a single list
        train = [item
                for sublist in copy_datasplit
                for item in sublist]
        #now add the 2016 data to the train list, but don't add records
        #that already exist
        # for record in train2:
        #     if record not in train:
        #         train.append(record)

        exdeflike = NER.trainExternal(train,
                                      multi)
        indeflike = NER.trainInternal(train)

        total_exdef[i] = exdeflike
        total_indef[i] = indeflike

    #write these dicts to the disk
    with open('exdeflike.txt', 'wb') as outfile:
        pickle.dump(total_exdef,
                  outfile)

    with open('indeflike.txt', 'wb') as f:
        pickle.dump(total_indef,
                  f)

def loadExtra():
    """this function will output the list _records_ of the supplementary
    internal model named entity data that we are using. here we will also
    pick and choose which of these files to apply the acronymization to"""

    raw_records = []
    #create a list of all the filenames in the lexicon data folder
    filenames = glob.glob('../data/lexicon/*.*')

    tuple_data = []
    for filename in filenames:
        with open(filename, 'r') as f:
            records = re.split("\n", f.read().strip())
            for record in records:
                data = record.split()
		sublist = []
                for i, token in enumerate(data):
                    if not i:
                        #if the word is first, it must have tag B
                        sublist.append((token, "B"))
                    else:
                        sublist.append((token, "I"))
		tuple_data.append(sublist)

    #now we load the add in the acronym data
    acronyms = glob.glob('../data/acronyms/*.*.*')

    for acronym in acronyms:
        with open(acronym, 'r') as filename:
            records = re.split("\n", filename.read().strip())
            for record in records:
		sublist = []
                if record:
                    token = record.split()[-1]   #we just want the acronym
                    sublist.append((token, "B")) #always a single word
		tuple_data.append(sublist)

    #train the write the data to disk
    NER.trainExtra(tuple_data)

## run crossTrain
## run crossValidate

def crossValidate(n = 10, multi = True,
                  threshold = [0.262, 0.880], NEtype = "location"):
    """perform n-fold cross validation on the data, in order to optimize
    the threshold parameters and maximize F1 score for the various models."""

    training_2016 = '../data/2016/data/train'
    training_2017 = '../data/2017/wnut17train.conll'

    data_split = split_data(training_2017, n)
    result = defaultdict(defaultdict)

    #load in the _exdeflike_ and _indeflike_ data
    with open('exdeflike.txt', 'rb') as infile:
        total_exdef = pickle.load(infile)

    with open('indeflike.txt', 'rb') as f:
        total_indef = pickle.load(f)

    exdeflikes = [None] * n
    indeflikes = [None] * n

    for fold in total_exdef:
        exdeflikes[int(fold)] = total_exdef[fold]
        indeflikes[int(fold)] = total_indef[fold]
        #put the indefs and exdefs in lists with the same indices

    for i, data in enumerate(data_split):
        #need to perform testing n times, giving each of the n partitions
        #of the data a chance to be the test data and the rest training
        duplicate = list(data_split)     #don't want to modify the original list
        test = duplicate.pop(i)
        #we split up runTest and evaluate
        print "performing " + str(i) +" fold"
        testData = NER.runTest(test,
                          exdeflikes[i],
                          indeflikes[i],
                          multi = multi,
                          NEtype = NEtype)
        result[str(i)] = NER.evaluate(testData,
                                      threshold,
                                      NEtype = NEtype)
    sort_result = sortDict(result)

    with open('2017plus_' + NEtype + '.result', 'w') as outfile:
        json.dump(sort_result,
                  outfile,
                  sort_keys = True,
                  indent = 4)
    return sort_result

def sortDict(result):
    """this rearranges the structure of _result_. _sort_result_ will be a
    dict with keys the param tuples (as strings still for the JSON module),
    values defaultdicts of lists where the keys are the statistical measures
    and values are lists of length n, where n is the number of folds"""

    sort_result = defaultdict(lambda: defaultdict(list))

    for fold in result:
        for param in result[fold]:
            sort_result[param]["F1"].append(result[fold][param]["F1"])
            sort_result[param]["fn"].append(result[fold][param]["fn"])
            sort_result[param]["fp"].append(result[fold][param]["fp"])
            sort_result[param]["precision"].append(result[fold][param]
                                                        ["precision"])
            sort_result[param]["recall"].append(result[fold][param]
                                                      ["recall"])
            sort_result[param]["tp"].append(result[fold][param]["tp"])

    return sort_result

def scanResult():
    """depending on the size of parameter scan the size of the JSON
    file produced above may be enormous. this function will allow us
    to examine the n choices of parameter with the highest F1 scores
    for each fold"""

    with open('2017plus.result', 'r') as f:
        result = json.load(f)

    data = [(sum(result[param]["F1"]) / 10, param)
	    for param in result]
    data = sorted(data)[::-1]
    return data

def trainFinal(
        multi = True,
        plus2016 = False,
        pluslexica = False,
        pluswiki = False,
        plusgmb = False,
        plusdev = False,
        plusconll2003 = False,
        doInternal = True,
        doExternal = True,
        lowercase = False,
        externalPOS = False,
        outhandle = ""
):
    """train all of the given training data and write it to disk for
    future use"""

    POS_train = []

    # load the 2017 data
    training_2017 = '../data/2017/wnut17train.conll'
    with open(training_2017, 'r') as f2:
        train = re.split("\n[\t]?\n", f2.read().strip())
        if externalPOS:
            for record in train:
                if record: #avoid empty strings
                    data = [re.split('\t', d)
                            for d in re.split("\n", record)
                            if len(re.split("\t", d)) == 2]
                    tokens, tags = zip(*data)
                    POSs = [t[1] for t in tagger.tag(tokens)]
                    POS_train.append("\n".join(["\t".join([POS, tag])
                                             for POS, tag in zip(POSs, tags)]))

    if plusconll2003:
        conll = "conll2003_toks.conll"
        with open(conll, 'r') as f:
            data_conll = re.split("\n[\t]?\n", f.read().strip())
        train.extend(data_conll)
        if externalPOS:
            conll = "conll2003_POSs.conll"
            with open(conll, 'r') as f:
                data_conll = re.split("\n[\t]?\n", f.read().strip())
            POS_train.extend(data_conll)

    if plusdev:
        dev = "../data/emerging.dev.conll"
        with open(dev, 'r') as f:
            data_dev = re.split("\n[\t]?\n", f.read().strip())
        train.extend(data_dev)
        if externalPOS:
            dev = "dev_POSs.conll"
            with open(dev, 'r') as f:
                data_dev = re.split("\n[\t]?\n", f.read().strip())
            POS_train.extend(data_dev)

    if pluswiki:
        wiki = "wiki_toks_amazing.conll"
        with open(wiki, 'r') as f:
            data_wiki = re.split("\n[\t]?\n", f.read().strip())
        train.extend(data_wiki)
        if externalPOS:
            wiki = "wiki_POSs.conll"
            with open(wiki, 'r') as f:
                data_wiki = re.split("\n[\t]?\n", f.read().strip())
            POS_train.extend(data_wiki)

    if plusgmb:
        gmb = "gmb_toks.conll"
        with open(gmb, 'r') as f:
            data_gmb = re.split("\n[\t]?\n", f.read().strip())
        train.extend(data_wiki)
        if externalPOS:
            gmb = "gmb_POSs.conll"
            with open(gmb, 'r') as f:
                data_gmb = re.split("\n[\t]?\n", f.read().strip())
            POS_train.extend(data_gmb)

    # load the 2016 data
    if plus2016:
        training_2016 = '../data/2016/data/train'
        dev_2016 = '../data/2016/data/dev'
        test_2016 = '../data/2016/data/test'

        with open(training_2016, 'r') as f:
            data_2016 = re.split("\n[\t]?\n", f.read().strip())
        with open(dev_2016, 'r') as f2:
            data_2016_dev = re.split("\n[\t]?\n", f2.read().strip())
        with open(test_2016, 'r') as f3:
            data_2016_test = re.split("\n[\t]?\n", f3.read().strip())
        train2 = data_2016 + data_2016_dev + data_2016_test

        train.extend(train2)

        if externalPOS:
            for record in train2:
                if record: #avoid empty strings
                    data = [re.split('\t', d)
                            for d in re.split("\n", record)
                            if len(re.split("\t", d)) == 2]
                    tokens, tags = zip(*data)
                    POSs = [t[1] for t in tagger.tag(tokens)]
                    POS_train.append("\n".join(["\t".join([POS, tag])
                                            for POS, tag in zip(POSs, tags)]))

    # load and weight the lexical data
    if pluslexica:
        numtoks = 0
        weights = []
        cts = Counter()
        NEcts = defaultdict(Counter)
        numNEs = Counter()
        totalNEs = 0
        for record in train:
            if record: #avoid empty strings
                data = [re.split('\t', d)
                        for d in re.split("\n", record)
                        if len(re.split("\t", d)) == 2]
                tokens, tags = zip(*data)
                if lowercase:
                    tokens = [token.lower() for token in tokens]
                for i, token in enumerate(tokens):
                    if tags[i] == "O":
                        cts[token] += 1
                    else:
                        NEcts[tags[i]][token] += 1
                        if tags[i][0] == "B":
                            totalNEs += 1
                            numNEs[tags[i]] += 1

            weights.append([1] * len(tokens))

        lexica = {
            ## "architecture.museum": "location",
            ## "automotive.make": "corporation",
            "automotive.model": "product",
            ## "broadcast.tv_channel": "corporation",
            ## "business.consumer_company": "corporation",
            "business.consumer_product": "product",
            ## "cvg.computer_videogame": "creative-work",
            ## "cvg.cvg_developer": "corporation",
            "cvg.cvg_platform": "product",
            "firstname.5k": "person", # note to Capcase this lexicon
            "lastname.5000": "person", # note to Capcase with B- and I-tags
            ## "location": "location",
            "location.country": "location",
            "people.family_name": "person", # note to Capcase with B- and I-tags
            "people.person.filtered": "person",
            ## "sports.sports_team": "group",
            ## "tv.tv_network": "corporation",
            ## "tv.tv_program": "creative-work",
        }

        lexNEs = Counter()
        lexrec = []
        lexweights = []
        lexterms = set()
        for lexicon in lexica:
            NEtype = lexica[lexicon]
            with open("/data/WNUT-NER-2017/data/lexicon/" + lexicon) as f:
                for line in f:
                    if lowercase:
                        line = line.strip().lower()
                    else:
                        line = line.strip()
                    tokens = [s
                              for s in re.split("([ \.\,])", line)
                              if s != " " and s]
                    if tokens:
                        lexNEs["B-" + NEtype] += 1
                        if not lowercase:
                            if (lexicon == "firstname.5k"
                                    or lexicon == "lastname.5000"
                                    or lexicon == "people.family_name"):
                                tokens = [s.capitalize()
                                          for s in re.split("([ \.\,])", line)
                                          if s != " " and s]

                        tags = ["I-" + NEtype for t in tokens]
                        tags[0] = "B-" + NEtype
                        for token, tag in zip(tokens, tags):
                            weight = 1
                            lexterms.add((token, tag, weight))
                        if (lexicon == "lastname.5000"
                                or lexicon == "people.family_name"):
                            weight = 1
                            lexterms.add((tokens[0], "I-" + NEtype, weight))

        for lexterm in lexterms:
            NEtype = "B" + lexterm[1][1:]
            lexweights.append([numNEs[NEtype] / lexNEs[NEtype]])
            record = "\t".join([lexterm[0], lexterm[1]])
            lexrec.append(record)

        for token in cts.most_common():
            lexweights.append([cts[token[0]]])
            record = "\t".join([token[0], "O"])
            lexrec.append(record)

        for NEtype in NEcts:
            if not lexNEs["B-"+NEtype[2:]]:
                for token in NEcts[NEtype].most_common():
                    lexweights.append([NEcts[NEtype][token[0]]])
                    record = "\t".join([token[0], NEtype])
                    lexrec.append(record)

    """
    #now load in the already trained extra data
    with open('extracon.txt', 'rb') as infile:
        con_counts = pickle.load(infile)
    with open('extradef.txt', 'rb') as f:
        deflike = pickle.load(f)
    """
    exdeflike = {}
    if doExternal:
        if externalPOS:
            exdeflike = NER.trainExternal(POS_train, multi)
        else:
            exdeflike = NER.trainExternal(train, multi)

    indeflike = {}
    if doInternal:
        if pluslexica:
            train.extend(lexrec)
            weights.extend(lexweights)
            indeflike = NER.trainInternal(train,
                                          weights = weights,
                                          lowercase = lowercase)
        else:
            indeflike = NER.trainInternal(train, lowercase = lowercase)

    #now write these dicts to disk
    exout = 'finalexdef_entitytype.pickle'
    inout = 'finalindef_entitytype.pickle'

    if externalPOS:
        exout = re.sub("_entitytype", "_externalPOS_entitytype" , exout)

    if outhandle:
        exout = re.sub("_entitytype", "_" + outhandle + "_entitytype" , exout)
        inout = re.sub("_entitytype", "_" + outhandle + "_entitytype" , inout)

    if plusdev:
        exout = re.sub("_entitytype", "_plusdev_entitytype" , exout)
        inout = re.sub("_entitytype", "_plusdev_entitytype" , inout)

    if plusconll2003:
        exout = re.sub("_entitytype", "_plusconll2003_entitytype" , exout)
        inout = re.sub("_entitytype", "_plusconll2003_entitytype" , inout)

    if plus2016:
        exout = re.sub("_entitytype", "_plus2016_entitytype" , exout)
        inout = re.sub("_entitytype", "_plus2016_entitytype" , inout)

    if pluslexica:
        exout = re.sub("_entitytype", "_pluslexica_entitytype" , exout)
        inout = re.sub("_entitytype", "_pluslexica_entitytype" , inout)

    if pluswiki:
        exout = re.sub("_entitytype", "_pluswiki_entitytype" , exout)
        inout = re.sub("_entitytype", "_pluswiki_entitytype" , inout)

    if plusgmb:
        exout = re.sub("_entitytype", "_plusgmb_entitytype" , exout)
        inout = re.sub("_entitytype", "_plusgmb_entitytype" , inout)

    if lowercase:
        exout = re.sub("_entitytype", "_lowercase_entitytype" , exout)
        inout = re.sub("_entitytype", "_lowercase_entitytype" , inout)

    if doExternal:
        with open(exout, 'wb') as outfile:
            pickle.dump(exdeflike, outfile)

    if doInternal:
        with open(inout, 'wb') as outfile2:
            pickle.dump(indeflike, outfile2)


def final_analysis(
        exdeflikefile, indeflikefile,
        multi = True, lowercase = True, externalPOS = True, dev = True,
        thresholds = {
            "location": 0.292,
            "group": 0.09,
            "product": 0.131,
            "creative-work": 1.1,
            "person": 0.202,
            "corporation": 1.1
        }):

    #load exdef and indef
    with open(exdeflikefile) as f:
        exdeflike = pickle.load(f)

    with open(indeflikefile) as f:
        indeflike = pickle.load(f)

    """train all of the given training data and then test it on the supplied
    test records. make predictions for NE for each token, then print them
    out in the format required"""

    """
    #load exdef and indef
    with open('finalexdef.pickle', 'rb') as infile:
        exdeflike = pickle.load(infile)

    with open('finalindef.pickle', 'rb') as infile2:
        indeflike = pickle.load(infile2)
    """
    #load the test data
    if dev:
        test_file = '../data/emerging.dev.conll'
        outfilename = "../data/finalpredictions/emerging_" + "_".join(
                        re.split("_", indeflikefile)[1:3]) + ".dev"
    else:
        test_file = '../data/emerging.test'
        outfilename = "../data/finalpredictions/emerging_" + "_".join(
                        re.split("_", indeflikefile)[1:3]) + ".test"

    with open(test_file, 'r') as f3:
        records = re.split("\n[\t]?\n", f3.read().strip())

    #analyze the test data
    #on the training data, using n-fold validation
    # f = open(test_file + ".prediction", 'w')
    f = open(outfilename, "w")

    # thresholds = {
    #     ## "location": [0.001, 0.157],
    #     "location": [1.1, 0.157],
    #     ## "group": [0.008, 0.199],
    #     "group": [1.1, 0.199],
    #     "product": [1.1, 0.215],
    #     "creative-work": [1.1,0.499],
    #     ## "person": [0.002, 0.167],
    #     "person": [1.1, 0.167],
    #     "corporation": [1.1, 0.218]
    # }

    for record in records:
        if record: #avoid empty strings
            if dev:
                data = [
                    re.split('\t', d)
                    for d in re.split("\n", record)
                    if len(re.split("\t", d)) == 2
                ]
                tokens, tags = zip(*data)
            else:
                tokens = [
                    re.split('\t', d)[0]
                    for d in re.split("\n", record)
                    if len(re.split("\t", d)) == 1
                ]
            uppertokens = list(tokens)
            if lowercase:
                tokens = [token.lower() for token in tokens]
            #keep track of the NE assignments for each token with tuples
            if lowercase:
                assignments = [[token, 'O']
                               for token in uppertokens
                               if token]
            else:
                assignments = [[token, 'O']
                                for token in tokens
                                if token]
            ##

            predictions = {}

            for NEtype in thresholds:
                exavedeflike, inavedeflike = NER.test(tokens,
                                                      exdeflike,
                                                      indeflike,
                                                      multi,
                                                      NEtype,
                                                      externalPOS = externalPOS,
                                                      uppertokens = uppertokens)

                #find the NEs using the _LFD_ function as before
                for indices in NER.LFD(tokens,
                                       exavedeflike,
                                       inavedeflike,
                                       [1.1, thresholds[NEtype]]):
                    # if exavedeflike[indices] >= thresholds[NEtype][0]:
                    #     print NEtype+": ", [tokens[ix] for ix in indices]
                    #     print "external", exavedeflike[indices]
                    innums = [inavedeflike[ix][1]
                              for ix in indices
                              if ix != indices[0]]
                    innums.append(inavedeflike[indices[0]][0])
                    ## print "internal", NER.harmonic_mean(innums)

                    predictions[(indices, NEtype)] = [len(list(indices)),
                                                      exavedeflike[indices],
                                                      NER.harmonic_mean(innums)]
            ##
            for indices, NEtype in predictions:
                thissize = predictions[(indices, NEtype)][0]
                thislike = predictions[(indices, NEtype)][2]
                for otherindices, otherNEtype in predictions:
                    thatsize = predictions[(otherindices, otherNEtype)][0]
                    thatlike = predictions[(otherindices, otherNEtype)][2]
                    broken = True
                    for ix in otherindices:
                        if ix in indices:
                            if otherindices[0] < indices[0]:
                                print("precidence, avoided: ",
                                      [tokens[ix] for ix in indices],
                                      " over ",
                                      [tokens[ix] for ix in otherindices])
                                break
                            elif otherindices[0] == indices[0]:
                                if thatsize > thissize:
                                    print("size, avoided: ",
                                          [tokens[ix] for ix in indices],
                                          " over ",
                                          [tokens[ix] for ix in otherindices])
                                    break
                                elif thatlike > thislike:
                                    print("likelihood, avoided: " + NEtype,
                                          [tokens[ix] for ix in indices],
                                          " over " + otherNEtype,
                                          [tokens[ix] for ix in otherindices])
                                    break
                    else:
                        broken = False
                    if broken:
                        break
                else:

                    print NEtype + ": ", [tokens[ix] for ix in indices]
                    print "internal", thislike
            ##
                    #assign 'B' to the first, 'I' to the rest
                    n = 0
                    for index in indices:
                        if n == 0:
                            assignments[index][1] = 'B-' + NEtype
                        else:
                            assignments[index][1] = 'I-' + NEtype
                        n += 1  #keep track of position in NE

            ##

            for i, assignment in enumerate(assignments):
                if dev:
                    f.writelines("\t".join([assignment[0],
                                    tags[i], assignment[1]]) + "\n")
                else:
                    f.writelines("\t".join([assignment[0],
                                            assignment[1]]) + "\n")

            f.writelines("\n")

def final_scan(exdeflike, indeflike, multi = True,
               lowercase = False, externalPOS = False, outkey = "default"):
    """train all of the given training data and then test it on the supplied
    test records. make predictions for NE for each token, then print them
    out in the format required"""

    """
    #load exdef and indef
    with open('finalexdef.pickle', 'rb') as infile:
        exdeflike = pickle.load(infile)

    with open('finalindef.pickle', 'rb') as infile2:
        indeflike = pickle.load(infile2)
    """
    #load the test data
    test_file = '../data/emerging.dev.conll'
    with open(test_file, 'r') as f3:
        records = re.split("\n[\t]?\n", f3.read().strip())

    numrecs = len(records)

    os.system("mkdir -p ../data/predictions/" + outkey)

    #analyze the test data
    # threshold = [0.138, 0.13]
    #this is the threshold we found to give the best F1 score
    #on the training data, using n-fold validation

    tstarts = {
        "location": 0.5,
        "group": 0.5,
        "product": 0.5,
        "creative-work": 0.5,
        "person": 0.5,
        "corporation": 0.5
    }

    tdiffs = range(-4, 5) # [d for d in range(-49,50)] ## diffs = [-0.49--0.49]
    allresults = defaultdict(list)

    ## begin rounds loop here
    for rnd in range(1, 4): ##range(1,2): ##
        print "working on rnd " + str(rnd)
        print "here are the starting thresholds: "
        for NEtype in tstarts:
            print NEtype, tstarts[NEtype]
        print
        results = defaultdict(list)
        tdiffs = [tdiff/10 for tdiff in tdiffs]
        # [tdiff/100 for tdiff in tdiffs] [-0.0049--0.0049]

        NEthreshs = {
            NEtype: [tstarts[NEtype] + tdiff for tdiff in tdiffs]
            # [t/1000 for t in range(1001)]
            for NEtype in tstarts
        }
        fs = {}
        for NEtype in NEthreshs:
            for t in NEthreshs[NEtype]:
                fkey = str(t)+"-"+NEtype
                threshfile = re.sub("/data/", "/data/predictions/" + outkey +
                                        "/", test_file + "-" + fkey +
                                        ".prediction")
                fs[fkey] = [open(threshfile, 'w'), threshfile]
                fs[fkey][0].close()
        numdone = 0
        for record in records:
            print str(100 * numdone / numrecs) + "% done with round " + str(rnd)
            numdone += 1
            if record: #avoid empty strings
                data = [re.split('\t', d)
                          for d in re.split("\n", record)
                          if len(re.split("\t", d)) == 2]
                tokens, tags = zip(*data)
                uppertokens = list(tokens)
                if lowercase:
                    tokens = [token.lower() for token in tokens]

                for NEtype in NEthreshs:
                    exavedeflike, inavedeflike = NER.test(tokens,
                                                    exdeflike,
                                                    indeflike,
                                                    multi,
                                                    NEtype,
                                                    externalPOS = externalPOS,
                                                    uppertokens = uppertokens)

                    for t in NEthreshs[NEtype]:
                        #keep track of the NE assignments
                        #for each token with tuples
                        if lowercase:
                            assignments = [[token, 'O']
                                           for token in uppertokens
                                           if token]
                        else:
                            assignments = [[token, 'O']
                                           for token in tokens
                                           if token]
                        fkey = str(t) + "-" + NEtype
                        #find the NEs using the _LFD_ function as before
                        for indices in NER.LFD(tokens,
                                               exavedeflike,
                                               inavedeflike,
                                               [1.1, t]):

                            # print t, [tokens[ix] for ix in indices]
                            # innums = [inavedeflike[ix][1] for ix in indices if ix != indices[0]]
                            # innums.append(inavedeflike[indices[0]][0])
                            # print "internal", NER.harmonic_mean(innums)
                            # raw_input()

                            n = 0
                            for index in indices:
                                if n == 0:
                                    assignments[index][1] = 'B-' + NEtype
                                else:
                                    assignments[index][1] = 'I-' + NEtype
                                n += 1  #keep track of position in NE

                        ## write out according to file handles, here
                        fs[fkey][0] = open(fs[fkey][1], "a")
                        for i, assignment in enumerate(assignments):
                            fs[fkey][0].writelines("\t".join([assignment[0],
                                            tags[i], assignment[1]]) + "\n")
                        fs[fkey][0].writelines("\n")
                        fs[fkey][0].close()

        ## evaluate all thresholds and all NE types for the best of the round
        for fkey in fs:
            ## fs[fkey][0].close()
            NEtype = "-".join(re.split("-", fkey)[1:])
            t = float(re.split("-", fkey)[0])
            filename = fs[fkey][1]
            try:
                results[NEtype].append(
                    (map(float,
                         re.split("\;",
                                  [re.sub("[^0-9\.\;]+", "", re.sub("\d+$|\d\:", "", r))
                                   for r in
                                   re.split("\n",
                                            subprocess.check_output(
                                                "python2 ../data/wnuteval.py "+filename,
                                                shell=True
                                            )
                                   )
                                   if re.search(NEtype, r)
                                  ][0]
                         )
                    ), t)
                )
            except:
                results[NEtype].append(([0.,0.,0.], t))
            allresults[NEtype].append(tuple(results[NEtype][-1]))

        ## store the best of this round as tstarts
        for NEtype in results:
            tstarts[NEtype] = max(results[NEtype], key = lambda x: x[0][2])[1]

        print "here are the end-of-round thresholds: "
        for NEtype in tstarts:
            print NEtype, tstarts[NEtype]
        print
    with open("../data/predictions/" + outkey + "/allresults.json", "w") as f:
        f.writelines(json.dumps([tstarts, allresults]))
    return tstarts, allresults
