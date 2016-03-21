#!/bin/sh
# this shell script removes the spaces out of the name of every file argument
# passed to it (handles an arbitrary number of files on the command line).
# example:
# remove_space_from_filename.sh "/Users/data/"
# new file will be saved without spaces in "/Users/data/"

for OriginalFile in $1/*;
do
    Location=`dirname "$OriginalFile"`
    FileName=`basename "$OriginalFile"`

    ShortName=`echo $FileName | sed 's/ //g'`

    if [ $ShortName != "$FileName" ]
    then
      cd "$Location"
      mv "$FileName" "$ShortName"
    fi
done