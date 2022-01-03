#!/bin/bash

for fname in $(cat $1);
do
  osf -p 8cy32 fetch -U "$fname" "$fname"
done
