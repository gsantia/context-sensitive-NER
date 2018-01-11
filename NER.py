#!/usr/bin/python
from __future__ import division
from collections import Counter, defaultdict
import re, sys, os, json, random, validate, glob
import numpy as np
import cPickle as pickle
from copy import deepcopy
from nltk.tag.perceptron import PerceptronTagger

tagger = PerceptronTagger()

def fqs(tokens, q, maxlength = 0):
    """finds the frequency of _tokens_ being split up into various subphrases,
    dependent on the probability that a space splits the phrase, _q_.
    _maxlength_ is a constraint on length of the NEs"""

    totalLength = len(tokens)

    #it's costly to look at all possible multi word named entities
    #when we know the vast majority of them will be a certain length
    #or shorter. _maxlength_ is our guess as to what this length should be
    if (not maxlength) or (maxlength > totalLength):
        maxlength = totalLength

    for length in range(1, maxlength + 1):
        #look at all possible subsets within _maxlength_
        for start in range(totalLength - length + 1):

            end = start + length
            pTokens = tokens[start: end] #the subphrase we're examining

            b = 0.
            if not start:   #phrase begins at the start of the sentence
                b += 1.
            if end == totalLength: #phrase ends at end of tokens
                b += 1.

            f = q ** (2. - b) * (1 - q) ** (length - 1.)

            #spit out a tuple of coordinates and prob of the phrase,
            #but only if prob is nonzero
            if f:
                yield [start, end] , f

def contexts(s_indices, tokens, tags, fs, numtok, q = 0.5):
    """allows for analysis of multi-word named entities. _s_indices_
    will slice up the initial tokens phrase. if we set q to 1 then
    we just get single word analysis since every space splits."""

    for t_indices, ft in fqs(tokens[s_indices[0]: s_indices[1]],
                             q,
                             maxlength = 5):
        #here we basically apply internal context analysis on the sliced tokens
        #phrase, _t_indices_ is the interval we will 'collapse' the context
        #into just a *
        context = list(tokens[s_indices[0]: s_indices[1]])
        tixs = range(s_indices[0], s_indices[1])[t_indices[0]: t_indices[1]]

        #this loop takes all the entries in range _t_indices_ and turns it into
        #a single "" entry in the list, representing a * as in the model
        for ix in range(t_indices[0], t_indices[1]):
            if ix == t_indices[0]:
                context[ix] = "" #the first item is made into ""
            else:
                context.pop(t_indices[0] + 1)   #and the rest are deleted

            f = (fs * ft * (s_indices[1] - s_indices[0]) *
                    (t_indices[1] - t_indices[0]))

            #now we need to check if the phrase in question is a named entity
            NE = False
            if re.search("^B[I]*$", "".join([tags[tix][0]
                         for tix in tixs
                         if tags[tix]])):
                if tixs[-1] == numtok - 1:
                    #if the named entity reaches the end of the phrase
                    NE = True
                elif tags[tixs[-1] + 1] != "I":
                    #otherwise, check if the first item AFTER isnt  NE
                    NE = True
        yield f, NE, tuple(context), tuple(range(t_indices[0], t_indices[1]))


def trainExternal(records, multi = True):
    """calculate the likelihoods for the provided records. we will be training
    by entity type as required"""

    #map the types onto the 2017 types
    entitytypes = {
        "corporation": "corporation",
        "company": "corporation",
        "creative-work": "creative-work",
        "tvshow": "creative-work",
        "movie": "creative-work",
        "group": "group",
        "sportsteam": "group",
        "location": "location",
        "geo-loc": "location",
        "facility": "location",
        "person": "person",
        "product": "product",
        "PER": "person",
        "LOC": "location",
        ## "ORG": "group",
        "geo": "location",
        "per": "person"## ,
        ## "gpe": "group",
        ## "org": "group"
    }

    ## patch deflike here to carry extra data (NER type)
    con_counts, deflike = Counter(), defaultdict(Counter)
    numrecords = 0
    print "working on external"
    poop = len(records)

    for record in records:
        if record:
            data, tokens, tags = munge(record)
            numtok = len(tokens)
            for s_indices, fs in fqs(tokens, 0.5):

                q = 0.5 if multi else 1
                #setting q=1 splits on every space, giving us the single case
                for f, NE, context, _ in contexts(s_indices,
                                                  tokens,
                                                  tags,
                                                  fs,
                                                  numtok,
                                                  q):
                    if NE: #update _deflike_ only for named entities
                        ## note patch here for entity types
                        if entitytypes.get(tags[0][2:], False):
                            entitytype = entitytypes[tags[0][2:]]
                            deflike[entitytype][context] += f
                        ## deflike["surface"][context] += f


                    con_counts[context] += f
        numrecords += 1
        print numrecords * 100 / poop

    ## note patch here for entity types
    for entitytype in deflike:
        deflike[entitytype] = Counter({context: deflike[entitytype][context]
                                      / con_counts[context]
                                      for context in deflike[entitytype]})
    return deflike

def internalContext(indices, token):
    """accept a token and indices and ouput tuple of the appropriate
    internal context"""

    context = list(token)
    numer = indices[1] - indices[0]
    for ix in range(indices[0], indices[1]):
        context[ix] = ""
    context = tuple(context)

    return context, numer

def munge(record, lowercase = False):
    """helper function which simply splits up a _record_ into a list
    of _data_ and then lists _tokens_ and _tags_"""
    data = [re.split("\t", d)
            for d in re.split("\n", record)
            if len(re.split("\t", d)) == 2]
    try:
        tokens, tags = zip(*data)
        if lowercase:
            tokens = [token.lower() for token in tokens]
        return data, tokens, tags
    except:
        print data

def trainInternal(records, weights = [], lowercase = False):
    """calculate the likelihoods for the training set using the
    internal context model."""

    print "working on internal"

    entitytypes = {
        "corporation": "corporation",
        "company": "corporation",
        "creative-work": "creative-work",
        "tvshow": "creative-work",
        "movie": "creative-work",
        "group": "group",
        "sportsteam": "group",
        "location": "location",
        "geo-loc": "location",
        "facility": "location",
        "person": "person",
        "product": "product",
        "PER": "person",
        "LOC": "location",
        ## "ORG": "group",
        "geo": "location",
        "per": "person"## ,
        ## "gpe": "group",
        ## "org": "group"
    }

    con_counts = Counter()
    deflike = defaultdict(Counter)

    poop = len(records)
    numrecords = 0

    for ri, record in enumerate(records): #split up the data as before
        if record:
            data, tokens, tags = munge(record, lowercase = lowercase)

            for i, token in enumerate(tokens):
                if weights:
                    weight = weights[ri][i]
                else:
                    weight = 1
                denom = len(token)
                for indices, f in fqs(token, 0.5):
                    #perform analysis on one word at a time
                    context, numer = internalContext(indices,
                                                     token)
                    if tags[i] != "O":  #only want the named entities
                        if entitytypes.get(tags[i][2:], False):
                            entitytype = tags[i][0]+"-"+entitytypes[tags[i][2:]]
                            deflike[entitytype][context] += (weight * f
                                                             * numer / denom)
                        ## deflike[tags[i]][context] += weight * f * numer / denom

                        ## deflike[tags[i][0]][context] += f * numer / denom

                        ## note patch below to record data beyond surface form
                        ## deflike[tags[i][2:]][context] += f * numer / denom

                        #need to normalize by word length
                        #we  set up a Counter for the first words in a NE and
                        #another for the non-first words in NEs
                    con_counts[context] += weight * f * numer / denom

        numrecords += 1
        print numrecords * 100 / poop

    for tag in deflike: #["I","B"]:
        deflike[tag] = Counter({context: deflike[tag][context] /
                                con_counts[context]
                                for context in deflike[tag]})
                                #if con_counts[context]})
        #perform division on each entry
    return deflike

def trainExtra(records):
    """to help with the internal model training we have several files of
    known named entities. this function will train them for the internal model
    and return the dicts/counters _con_counts_ and _deflike_ for the
    _trainInternal_ function to use"""

    print "working on extra stuff"
    con_counts = Counter()
    deflike = defaultdict(Counter)

    for record in records:
	if record:
            tokens, tags = zip(*record)
            for i, token in enumerate(tokens):
                denom = len(token)
                for indices, f in fqs(token, 0.5):
                    context, numer = internalContext(indices,
                                                 token)
                    #here we know everything is a NE
                    deflike[tags[i]][context] += f * numer / denom
                    con_counts[context] += f * numer / denom
    #we don't need to do normalization here because it'll be done later
    #at the end of _trainExternal_, since we're just passing the dicts
    with open('extracon.txt', 'wb') as outfile:
        pickle.dump(con_counts, outfile)

    with open('extradef.txt', 'wb') as f:
        pickle.dump(deflike, f)


def newInternal(word, deflike, NEtype = None):
    """perform internal context analysis on arbitrary text input, also needs
    the deflike dictionary to be created already"""

    avedeflike = [0,0]
    denom = len(word)
    for indices, f in fqs(word, 0.5):
        context, numer = internalContext(indices, word)
        for i, tag in enumerate(["B", "I"]):
            index = tag + "-" + NEtype
            if deflike[index][context]:
                avedeflike[i] += numer/denom * f * deflike[index][context]
    return avedeflike


def newExternal(tokens, exdeflike, multi = True, NEtype = None):
    """perform external context analysis on arbitrary text input, also needs
    the deflike dictionary to be created already"""
    numtok = len(tokens)
    avedeflike = Counter() # defaultdict(Counter)
    q = 0.5 if multi else 1 #again if we set q=1 then we split on every space,
                            #making all NEs single word
    for s_indices, fs in fqs(tokens, 0.5):
        for f, NE, context, t_indices in contexts(s_indices,
                                                  tokens,
                                                  ["O" for token in tokens],
                                                  fs,
                                                  numtok,
                                                  q):
            s_range = range(s_indices[0], s_indices[1])
            t_indices = tuple(s_range[t]
                              for t in t_indices)
            #t_indices refers to the index inside the phrase,
            #we need to place it in the right context
        #for i, context in oldContexts(indices, tokens):
            # for entitytype in exdeflike:
            if exdeflike[NEtype][context] != None:
                avedeflike[t_indices] += f * exdeflike[NEtype][context]

    return avedeflike

def test(tokens, exdeflike, indeflike, multi = True,
         NEtype = None, externalPOS = False, uppertokens = []):
    """perform both analyses on a list of strings. allows for multi word
    named entities by putting multi = True"""

    if not uppertokens:
        upptertokens = tokens

    ## note, if we want to use exdeflike beyond surface,
    ## this is where the modified should be engaged
    ## e.g., pass exdeflike["surface"] to distinguish from entity types
    ## indef checks out and can carry the extra data benignly
    if externalPOS:
        exavedeflike = newExternal([t[1] for t in tagger.tag(uppertokens)],
                                   exdeflike,
                                   multi,
                                   NEtype)
    else:
        exavedeflike = newExternal(tokens,
                                   exdeflike,
                                   multi,
                                   NEtype)
    inavedeflike = []
    for i, token in enumerate(tokens):
        #each token has different likelihoods for being the beginning of
        #a named entity "B" or somewhere in the middle "I". we store both
        #here in _inavedeflike_
        ## inavedeflike.append([newInternal(token, indeflike["B"]),
                            ## newInternal(token, indeflike["I"])])
        inavedeflike.append(newInternal(token, indeflike, NEtype))

    return exavedeflike, inavedeflike

def harmonic_mean(numbers):
    """take the harmonic mean of a list of numbers"""
    if min(numbers) == 0:
        return 0
    else:
        denom = 0.0
        for num in numbers:
            denom += 1.0 / num

        return len(numbers) / denom

def LFD(tokens, exavedeflike, inavedeflike, threshold):
    """this is a method that looks for dictionary definitions in the
    possible subsets of tokens. Longest First Defined"""

    allidx = range(len(tokens))
    while len(tokens):
        ixs = reversed(range(1, len(tokens) + 1))
        #we want this list to be reverse order
        for ix in ixs:
            indices = tuple(allidx[0:ix])
            #check if the subphrase created by these indices is above the given
            #threshold or if ix is 1, so the subphrase is just a single word
            el = exavedeflike[indices]
            il = [inavedeflike[x][1]
                  for x in indices]
            #each token has different internal likelihood
            il[0] = inavedeflike[indices[0]][0]
            #only the first token is a "B"
            #now take the harmonic mean to reduce to a single value
            il = harmonic_mean(il)

            if decide(el, il, threshold) or (not ix - 1):
                #we check if the phrase meets our condition
                if decide(el, il, threshold):
                    #this is for if ix != 1
                    yield indices
                #if we have just a single word, delete it
                if len(tokens) == 1:
                    tokens = []
                    allidx = []
                #here we chop off the rest of the tokens for analysis
                else:
                    tokens = list(tokens[ix:])
                    allidx = list(allidx[ix:])
                break

def decide(el, il, threshold):
    """returns boolean saying if we meet the chosen parameters for
    just the harmonic model. it's by far the best for F1 score"""
    result = (el >= threshold[0]) or (il >= threshold[1])
    return result

def getPositives(data, NEtype = None):
    """looks through the given _data_ set for named entities marked as such.
    returns the set _positives_"""

    positives = set()
    NE = []
    for i, (token, tag) in enumerate(data):
        if tag == "B-" + NEtype:   #the start of a named entity
            if NE:
                #in this case we have two named entities next to each other,
                #so need to add the previous NE to _positive_ and then delete it
                positives.add(tuple(NE))
                NE = []
            else:
                #the start of a new NE
                NE.append(i)
        elif tag == "O":
            if NE:
                #this is the end of the current NE, so add it to _positives_
                #and clear NE
                positives.add(tuple(NE))
                NE = []
        elif tag == "I-" + NEtype:
            #here we are in the middle of the NE, so just add to the list
            NE.append(i)
    if NE:
        #if once we finish with the above loop and NE is nonempty, it means we
        #ended on a word in a NE, so make sure to add it to _positives_
        positives.add(tuple(NE))

    positives = {NE
                for NE in positives
                if len(NE) == 1}
    return positives

def runTest(records, exdeflike, indeflike, multi = True, NEtype = None):
    """takes the trained data and applies it to the test data, returns the
    list of dicts _testData_ which stores for each _record_ the record,
    tokens, positives, exavedeflike, and inavedeflike"""

    testData = []

    for record in records:
        testDict = {"record": "",
                    "tokens": [],
                    "positives": set(),
                    "exavedeflike": Counter(),
                    "inavedeflike": []}

        if record:  #this section is just to catch pathologies
            testDict["record"] = record
            data = [re.split("\t", d)
                    for d in re.split("\n", record)
                    if len(re.split('\t', d)) == 2]
            try:
                testDict["tokens"], tags = zip(*data)
            except:
                print data

            positives = getPositives(data, NEtype)
            testDict["positives"] = positives

            #create a copy of the positives set for each model
            ## for inavedeflike we now have a list of dictionaries,
            ## each rating the tag possibilities by token
            testDict["exavedeflike"], testDict["inavedeflike"] = test(
                                                    testDict["tokens"],
                                                    exdeflike,
                                                    indeflike,
                                                    multi,
                                                    NEtype)
            testData.append(testDict)
    return testData

def evaluate(testdata, threshold, NEtype = None):
    """given _testData_ and a _threshold_ value calculates the results
    of the test including tp, fp, tn, fn, and the various statistical
    measures we'd like. returns dict _ev_ with all this data."""

    #parameter scan
    param = defaultdict(lambda: defaultdict(float))

    t1s = [1./1000 * n for n in range(1001)]
    t2s = t1s

    #first we want to scan over 1000 choices for t1
    for t1 in t1s:
        threshold = [t1, 1.0]
        print "scanning at", threshold
        ev = evaluate_helper(threshold,
                             testdata,
                             NEtype)
        param[str((t1, 1.0))] = ev

    #next scan over 1000 choices for t2
    for t2 in t2s:
        threshold = [1.0, t2]
        print "scanning at", threshold
        ev = evaluate_helper(threshold,
                             testdata,
                             NEtype)
        param[str((1.0, t2))] = ev
    """
    for t1 in t1s:
        for t2 in t2s:
            threshold = [t1, t2]
            print "scanning at", threshold
            ev = evaluate_helper(threshold, testdata)
            param[str((t1, t2))] = ev
    """
    return param

def evaluate_helper(threshold, testdata, NEtype = None):
    """assigns the values "tp", "fp", and "fn" and then performs
    the calculations to get the statistical measures we want and
    returns them in a dict called _ev_"""

    ev = defaultdict(float)
    for testdict in testdata:
        positives = deepcopy(testdict["positives"])

        for indices in LFD(testdict["tokens"],
                           testdict["exavedeflike"], #testdict["exavedeflike"][NEtype],
                           testdict["inavedeflike"], #[[d["B-"+NEtype],d["I-"+NEtype]] for d in testdict["inavedeflike"]],
                           threshold):
            if indices in positives:
                #here we have a defined ne
                result = "tp"
                positives.remove(indices)
                #don't want to repeat the ne
            else:
                result = "fp"

            ev[result] += 1

        ev["fn"] += len(positives)

    #need to assign the statistical measures a value of 0
    #in the case of dividing by 0
    if ev["tp"] + ev["fp"] <= 0:
        ev["precision"] = 0
    else:
        ev["precision"] = ev["tp"] / (ev["tp"] + ev["fp"])

    if ev["tp"] + ev["fn"] <= 0:
        ev["recall"] = 0
    else:
        ev["recall"] = ev["tp"] / (ev["tp"] + ev["fn"])

    ev["F1"] = harmonic_mean([ev["precision"], ev["recall"]])
    return ev




