#!/bin/bash

# Merge course files
cat files_tsv/course1.tsv > merged_courses.tsv
tail -q -n +2 files_tsv/course{2..6000}.tsv >> merged_courses.tsv

# Which country offers the most Master's Degrees?
csvcut -t -c country merged_courses.tsv | sort | uniq -c | sort -nr > country_count_sorted.txt
echo "Country with the largest Master Course offering:"
head -n 1 country_count_sorted.txt | awk '{print $2}'

# Which city?
csvcut -t -c city merged_courses.tsv | sort | uniq -c | sort -nr > city_count_sorted.txt
echo "City with the largest Master Course offering:"
head -n 1 city_count_sorted.txt | awk '{print $2}'

# How many colleges offer Part-Time education?
csvcut -t -c isItFullTime merged_courses.tsv | grep -i -E "part\s*time" | wc -l > part_time_count.txt
echo "Part time count:"
cat part_time_count.txt

# Print the percentage of courses in Engineering

engineer_count=$(csvcut -t -c courseName merged_courses.tsv | grep -iE "engineer" | wc -l)
len=$(wc -l < merged_courses.tsv)
echo "Percentage of courses with 'Engineering' in title (course name):"
echo "$(echo "scale=2; ($engineer_count / $len) * 100" | bc)%"

