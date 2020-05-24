#!/bin/bash

out=$(cat)

echo $out | tr '\r' '\n' | grep -oP 'acc: \K[0-9]*.[0-9]*' | nl
