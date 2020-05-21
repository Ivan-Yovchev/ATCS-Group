#!/bin/bash

tr '\r' '\n' < $1 | grep -oP 'acc: \K[0-9]*.[0-9]*' | nl
