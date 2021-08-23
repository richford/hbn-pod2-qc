#!/bin/bash

firebase login
PROJECTS=$(firebase projects:list | tail -n+4 | sed -n 'p;n' | sed 's/│/|/g' | cut -d '|' -f 2)

PS3='Please enter a number from above list: '
echo "Select the firebase project from which to download your data:"

select proj in $PROJECTS
do
    echo "Downloading data from project $proj"
    firebase database:get --project $proj -o /output/$proj.json /
    break
done
