#!/bin/bash

out=$(cat)

echo 'Per epoch accuracies:'
echo $out | tr '\r' '\n' | grep -oP 'acc: \K[0-9]*.[0-9]*' | nl

echo 'Final accuracy:'
echo $out | tr '\r' '\n' | grep -oP 'Final: \K[0-9]*.[0-9]*'
