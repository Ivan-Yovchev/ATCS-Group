#!/bin/bash

cat | tr '\r' '\n' | grep -oP 'acc: \K[0-9]*.[0-9]*' | nl
