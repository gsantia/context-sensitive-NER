#!/usr/bin/python

import NER

execfile("validate.py")

crossValidate(n = 10, multi = True)

