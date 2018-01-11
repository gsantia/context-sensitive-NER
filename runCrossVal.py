#!/usr/bin/python

## simply run this script as one of the six:
##
## python2 runCrossVal.py location/person/corporation/creative-work/group/product


import NER, sys

execfile("validate.py")

NEtype = sys.argv[1]

crossValidate(NEtype = NEtype)
