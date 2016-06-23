#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
# classify text

import sentiment_mod as s

print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
print(s.sentiment("very bad movie"))
print(s.sentiment("very good"))
print(s.sentiment("this is a very enjoyable movie. you will leave it feeling good, and maybe thinking a bit as well. think dolphin tale. wish they had the ski jump sceens in imax"))
print(s.sentiment("Inspiring story. A bit hollywood'ish. Moves right along with good acting. Our entire family enjoyed it. "))

