#!/usr/bin/python
from __future__ import division
from collections import Counter, defaultdict
import re, sys, os, json
import numpy as np
import random
import validate


def fqs(tokens, q):
    """finds the frequency of _tokens_ being split up into various subphrases,
    dependent on the probability that a space splits the phrase, _q_"""

    totalLength = len(tokens)
    for length in range(1, totalLength + 1):    #need to look at all possible subsets of tokens
        for start in range(totalLength - length + 1):

            end = start + length
            pTokens = tokens[start:end] #the subphrase we're examining

            b = 0.
            if not start:   #phrase begins at the start of the sentence
                b += 1.
            if end == totalLength: #phrase ends at end of tokens
                b += 1.

            f = q**(2. - b)*(1 - q)**(length - 1.)  #this is our formulation for probability

            #spit out a tuple of coordinates and prob of the phrase, but only if prob is nonzero
            if f:
                yield [start, end] , f


def oldContexts(indices, tokens):
    """yields the various possible external contexts for words to appear in the given tokens.
    the project focuses on single word named entities, so we only need contexts with one *"""

    pTokens = tokens[indices[0]:indices[1]]
    ixs = range(indices[0], indices[1])
    for i, token in enumerate(pTokens):
        context = list(pTokens)
        context[i] = ""     #this will represent the * as described in the model
        yield ixs[i], tuple(context)


def contexts(s_indices, tokens, tags, fs, numtok, q = 0.5):
    """expanded version of the _oldContexts_ function. allows for analysis of multi-word
    named entities. _s_indices_ will slice up the initial tokens phrase"""

    for t_indices, ft in fqs(tokens[s_indices[0]:s_indices[1]], q):
        #here we basically apply internal context analysis on the sliced tokens phrase,
        #_t_indices_ is the interval we will 'collapse' the context into just a *
        context = list(tokens[s_indices[0]:s_indices[1]])   #create a new list to work with
        tixs = range(s_indices[0], s_indices[1])[t_indices[0]:t_indices[1]]

        #this loop takes all the entries in range _t_indices_ and turns it into a single
        #"" entry in the list, representing a * as in the model
        for ix in range(t_indices[0], t_indices[1]):
            if ix == t_indices[0]:
                context[ix] = "" #the first item is made into ""
            else:
                context.pop(t_indices[0] + 1)   #and the rest are deleted

            f = fs * ft * (s_indices[1] - s_indices[0]) * (t_indices[1] - t_indices[0])

            #now we need to check if the phrase in question is a named entity
            NE = False
            if re.search("^B[I]*$", "".join([tags[tix] for tix in tixs])):
                if tixs[-1] == numtok - 1:  #if the named entity reaches the end of the phrase
                    NE = True
                elif tags[tixs[-1] + 1] != "I": #otherwise, check if the first item AFTER the named
                    NE = True                   #entity is NOT a named entity

        yield f, NE, tuple(context)




def trainExternal(record, multi = False):
    """calculate the likelihoods for the training set. for now, we just want to examine
    single word named entities, so multi = False. but the function can be used for multi
    by indicating multi = True"""

    con_counts = Counter()
    deflike = Counter()

    for record in records:
        data = [re.split("\t", d) for d in re.split("\n", record)]
        tokens, tags = zip(*data)
        numtok = len(tokens)
        for s_indices, fs in fqs(tokens, 0.5):     #for now, say q is 50%

            if multi:   #if the user wishes to analyze multi word named entities
                for f, NE, context in contexts(s_indices, tokens, tags, fs, numtok, 0.5):
                    if NE:
                        deflike[context] += f
                    con_counts[context] += f

            else:   #otherwise perform single word named analysis
                for ixs, context in oldContexts(s_indices, tokens):
                    if fs:   #only add to the deflike dict when the likelihood is nonzero
                        deflike[context] += fs
                    con_counts[context] += fs

    deflike = Counter({context: deflike[context]/con_counts[context] for context in deflike})   #perform division on each entry

    return deflike

def internalContext(indices, token):
    """accept a token and indices and ouput tuple of the appropriate internal context"""

    context = list(token)
    numer = indices[1] - indices[0]
    for ix in range(indices[0], indices[1]):
        context[ix] = ""
    context = tuple(context)

    return context, numer


def trainInternal():
    """calculate the likelihoods for the training set using the
    internal context model"""

    con_counts = Counter()
    deflike = Counter()

    for record in records:
        data = [re.split("\t", d) for d in re.split("\n", record)]
        tokens, tags = zip(*data)

        for i, token in enumerate(tokens):
            denom = len(token)
            for indices, f in fqs(token, 0.5):  #perform analysis on one word at a time
                context, numer = internalContext(indices, token)
                if tags[i] != "O":  #only want the named entities
                    deflike[context] += f * numer/denom #need to normalize by word length
                con_counts[context] += f * numer/denom

    deflike = Counter({context: deflike[context]/con_counts[context] for context in deflike})   #perform division on each entry

    return deflike


def testInternal():
    """assign likelihoods to tokens based on the training data, using internal context model"""

    deflike = trainInternalLikelihood()     #perform the training

    token_counts = Counter()
    avedeflike = Counter()

    with open ("../data/2016/data/test", "r") as f:
        records = re.split("\n\n",f.read().strip())     #separate by double new line

    for record in records:
        data = [re.split("\t", d) for d in re.split("\n", record)]
        try:
            tokens, tags = zip(*data)
        except:
            print data
            pass

        for token in tokens:
            token_counts[token] += 1.
            denom = len(token)
            for indices, f in fqs(token, 0.5):
                context, numer = internalContext(indices, token)
                if deflike[context]:
                    avedeflike[token] += numer/denom * f * deflike[context]

    avedeflike = Counter({token: avedeflike[token] / token_counts[token] for token in avedeflike})
    return avedeflike

def newInternal(word, deflike):
    """perform internal context analysis on arbitrary text input, also needs
    the deflike dictionary to be created already"""

    avedeflike = 0
    denom = len(word)
    for indices, f in fqs(word, 0.5):
        context, numer = internalContext(indices, word)
        if deflike[context]:
            avedeflike += numer/denom * f * deflike[context]
    return avedeflike


def testExternal(multi = False):
    """assign likelihoods to tokens based on the training data. again allows the user
    to set multi = True to switch from single word named entitiy analysis to multi"""

    deflike = trainExternalLikelihood(multi)     #perform the training
    token_counts = Counter()
    avedeflike = Counter()

    with open ("../data/2016/data/test", "r") as f:
        records = re.split("\n\n", f.read().strip())     #separate by double new line

    for record in records:
        data = [re.split("\t", d) for d in re.split("\n", record)]
        try:
            tokens, tags = zip(*data)
        except:
            print data
            pass

        numtok = 0

        for token in tokens:
            token_counts[token] += 1.
            numtok += 1

        for indices, f in fqs(tokens, 0.5):
            if multi:   #multi word named entity analysis
                for f, NE, context in contexts(indices, tokens, tags, f, numtok, 0.5):
                    if deflike[context]:
                        avedeflike[tokens[i]] += f * deflike[context]


            else:       #single word named entity analysis
                for i, context in oldContexts(indices, tokens):
                    if deflike[context]:
                        avedeflike[tokens[i]] += f * deflike[context]

    avedeflike = Counter({token: avedeflike[token] / token_counts[token] for token in avedeflike})

    return avedeflike

def newExternal(tokens, deflike):
    """perform external context analysis on arbitrary text input, also needs
    the deflike dictionary to be created already"""

    avedeflike = Counter()
    for indices, f in fqs(tokens, 0.5):
        for i, context in oldContexts(indices, tokens):
            if deflike[context]:
                avedeflike[i] += f * deflike[context]

    return avedeflike

def test(tokens, exdeflike, indeflike):
    """perform both analyses on a given string"""

    avedeflike = newExternal(tokens, exdeflike)
    for i, token in enumerate(tokens):
        yield i, token, avedeflike[i], newInternal(token, indeflike)

def harmonic_mean(numbers):
    """take the harmonic mean of a list of two numbers"""
    return 2 * numbers[0] * numbers[1] / sum(numbers)

def decide(el, il, model, threshold):
    """returns boolean saying if we meet the chosen parameters for
    each model. the 4th model is just random choice, so we find a
    random decimal and round"""

    if model == 0:
        return el >= threshold[0] and il >=threshold[1]
    elif model == 1:
        return el >= threshold[0] or il >= threshold[1]
    elif model == 2:
        return harmonic_mean([el, il]) >= harmonic_mean(threshold)
    else:
        return bool(round(random.random()))


def runTest(exdeflike, indeflike):
    """takes the trained data and applies it to the test data, returns a dict
    which has statistical measure as keys and a list of those measures for the
    four different models as the values."""

    with open ("../data/2016/data/test", "r") as f:
        records = re.split("\n\n", f.read().strip())     #separate by double new line

    threshold = [0.3, .1]  #just a guess for now
    ev = defaultdict(lambda: [0,0,0,0])

    for record in records:
        data = [re.split("\t", d) for d in re.split("\n", record)]
        try:
            tokens, tags = zip(*data)
        except:
            print data
            pass

        for i, token, el, il in test(tokens, exdeflike, indeflike):
            for model in range(4):
                result = "tn"
                if decide(el, il, model, threshold):
                    result = "tp" if tags[i][0] == "B" else "fp"
                elif tags[i][0] == "B":
                    result = "fn"
                ev[result][model] += 1

    for model in range(4):
        ev["precision"][model] = ev["tp"][model] / (ev["tp"][model] + ev["fp"][model])
        ev["recall"][model] = ev["tp"][model] / (ev["tp"][model] + ev["fn"][model])
        ev["F1"][model] = harmonic_mean([ev["precision"][model], ev["recall"][model]])

    return ev

def validate(n = 5):
    """perform n-fold cross validation on the data, in order to optimize the threshold
    parameters and maximize F1 score for the various models"""


#==========================================================================================================================
# IMPLEMENTATION
#==========================================================================================================================

if __name__ == "__main__":

    with open ("../data/2016/data/train", "r") as f:
        records = re.split("\n\n", f.read().strip())

    exdeflike = trainExternal(records)


    with open ("../data/2016/data/train", "r") as f:
        records = re.split("\n\n", f.read().strip())

    indeflike = trainInternal(records)



# TODO: n-fold cross validation, n = 5? function of n
# break up data into n equal-sized chunks
# for given fold, train on other n-1, then test on the fold
# split training set into n pieces, optimize F1
# scipy.optimize minimize -f1
# adjust threshold, give the optimizer a dict with record, el, and il
# TODO:
# make work for multi-word named entities
#
