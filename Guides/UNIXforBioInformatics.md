# Unix Command Guide: From Basics to Bioinformatics

## Table of Contents
1. [Basic Unix Commands](#basic-unix-commands)
2. [File Operations](#file-operations)
3. [Directory Operations](#directory-operations)
4. [File Viewing and Editing](#file-viewing-and-editing)
5. [File Permissions](#file-permissions)
6. [Process Management](#process-management)
7. [System Information](#system-information)
8. [Text Processing](#text-processing)
9. [Finding Files and Content](#finding-files-and-content)
10. [Compression and Archives](#compression-and-archives)
11. [Networking Commands](#networking-commands)
12. [Disk Usage](#disk-usage)
13. [SSH and Remote Access](#ssh-and-remote-access)
14. [Shell Scripting Basics](#shell-scripting-basics)
15. [Bioinformatics Tools](#bioinformatics-tools)
    - [Sequence Analysis](#sequence-analysis)
    - [Alignment Tools](#alignment-tools)
    - [Assembly Tools](#assembly-tools)
    - [Variant Calling](#variant-calling)
    - [RNA-Seq Analysis](#rna-seq-analysis)
    - [Metagenomics](#metagenomics)
    - [Visualization](#visualization)

## Basic Unix Commands

### `pwd` - Print Working Directory
```bash
pwd
# Output: /home/username/documents
```

### `ls` - List Directory Contents
```bash
# Basic listing
ls

# Long format with details
ls -l

# Show hidden files
ls -a

# Human-readable file sizes
ls -lh

# Sort by modification time
ls -lt

# Recursive listing
ls -R
```

### `cd` - Change Directory
```bash
# Go to home directory
cd

# Go to specific directory
cd /path/to/directory

# Go up one directory
cd ..

# Go to previous directory
cd -

# Go to home directory (alternative)
cd ~
```

### `man` - Manual Pages
```bash
# View manual for a command
man ls
```

### `help` - Command Help
```bash
# Get help for bash built-in commands
help cd
```

### `history` - Command History
```bash
# View command history
history

# Run command number 42 from history
!42

# Run last command
!!
```

### `clear` - Clear Terminal
```bash
clear
```

### `echo` - Display Text
```bash
# Print text
echo "Hello World"

# Print variable
echo $PATH
```

### `date` - Display Date and Time
```bash
date
# Output: Mon Mar 3 14:23:45 PST 2025
```

### `cal` - Display Calendar
```bash
# Current month
cal

# Specific year
cal 2025
```

## File Operations

### `touch` - Create Empty File
```bash
# Create new file
touch filename.txt

# Update timestamp of existing file
touch -a filename.txt
```

### `cp` - Copy Files
```bash
# Copy file
cp file1.txt file2.txt

# Copy directory and its contents
cp -r directory1 directory2

# Preserve attributes
cp -p file1.txt file2.txt

# Interactive mode (prompt before overwrite)
cp -i file1.txt file2.txt
```

### `mv` - Move/Rename Files
```bash
# Move file to directory
mv file.txt directory/

# Rename file
mv oldname.txt newname.txt

# Interactive mode
mv -i file1.txt file2.txt
```

### `rm` - Remove Files
```bash
# Remove file
rm file.txt

# Remove directory and contents
rm -r directory/

# Force removal
rm -f file.txt

# Interactive mode
rm -i file.txt
```

### `ln` - Create Links
```bash
# Create hard link
ln file1.txt file2.txt

# Create symbolic link
ln -s file1.txt file2.txt
```

## Directory Operations

### `mkdir` - Make Directory
```bash
# Create directory
mkdir directory

# Create parent directories if needed
mkdir -p path/to/new/directory

# Set mode (permissions)
mkdir -m 755 directory
```

### `rmdir` - Remove Directory
```bash
# Remove empty directory
rmdir directory

# Remove multiple directories
rmdir dir1 dir2 dir3
```

## File Viewing and Editing

### `cat` - Concatenate and Display Files
```bash
# Display file contents
cat file.txt

# Display multiple files
cat file1.txt file2.txt

# Display with line numbers
cat -n file.txt

# Create file with content
cat > newfile.txt
This is content
Press Ctrl+D to save and exit
```

### `less` - View Files Page by Page
```bash
less file.txt
# Navigation: Space (next page), b (previous page), q (quit)
```

### `more` - View Files (Older Version of less)
```bash
more file.txt
```

### `head` - Display First Lines of File
```bash
# Display first 10 lines
head file.txt

# Display first n lines
head -n 20 file.txt
```

### `tail` - Display Last Lines of File
```bash
# Display last 10 lines
tail file.txt

# Display last n lines
tail -n 20 file.txt

# Follow file (show new content as it's added)
tail -f log.txt
```

### `nano` - Simple Text Editor
```bash
nano file.txt
```

### `vim` - Advanced Text Editor
```bash
vim file.txt
# Press i to enter insert mode
# Press Esc to exit insert mode
# Type :wq to save and quit
# Type :q! to quit without saving
```

## File Permissions

### `chmod` - Change File Mode/Permissions
```bash
# Make file executable
chmod +x script.sh

# Set specific permissions (rwx for user, rx for group/others)
chmod 755 file.txt

# Recursive permission change
chmod -R 755 directory/

# Using symbolic notation
chmod u=rwx,g=rx,o=rx file.txt
```

### `chown` - Change File Owner
```bash
# Change owner
chown user file.txt

# Change owner and group
chown user:group file.txt

# Recursive ownership change
chown -R user:group directory/
```

### `chgrp` - Change Group Ownership
```bash
chgrp group file.txt
```

## Process Management

### `ps` - Process Status
```bash
# View your processes
ps

# View all processes
ps -e

# Detailed process info
ps -ef

# Format output
ps -eo pid,user,command
```

### `top` - Dynamic Process Viewer
```bash
top
# Press q to quit
```

### `htop` - Enhanced Process Viewer
```bash
htop
# More user-friendly than top
```

### `kill` - Terminate Process
```bash
# Kill by process ID
kill 1234

# Force kill
kill -9 1234

# Kill processes by name
killall firefox
```

### `bg` - Background Jobs
```bash
# Move job to background
bg %1
```

### `fg` - Foreground Jobs
```bash
# Move job to foreground
fg %1
```

### `jobs` - List Background Jobs
```bash
jobs
```

### `nohup` - Run Command Immune to Hangups
```bash
nohup long_running_command &
```

### `nice` - Run Command with Modified Priority
```bash
nice -n 10 command
```

## System Information

### `uname` - System Information
```bash
# All system info
uname -a

# Kernel name
uname -s

# Kernel version
uname -r

# Hardware name
uname -m
```

### `df` - Disk Free Space
```bash
# Show disk usage
df

# Human-readable format
df -h
```

### `free` - Memory Usage
```bash
# Show memory usage
free

# Human-readable format
free -h
```

### `uptime` - System Uptime
```bash
uptime
```

### `who` - Users Currently Logged In
```bash
who
```

### `whoami` - Current User
```bash
whoami
```

### `w` - Who is Logged In and What They're Doing
```bash
w
```

## Text Processing

### `grep` - Search Text
```bash
# Search for pattern
grep "pattern" file.txt

# Case-insensitive search
grep -i "pattern" file.txt

# Recursive search in directory
grep -r "pattern" directory/

# Show line numbers
grep -n "pattern" file.txt

# Count matches
grep -c "pattern" file.txt

# Show only matching part
grep -o "pattern" file.txt

# Extended regex
grep -E "pattern1|pattern2" file.txt
```

### `sed` - Stream Editor
```bash
# Replace first occurrence
sed 's/old/new/' file.txt

# Replace all occurrences
sed 's/old/new/g' file.txt

# Replace on specific line
sed '3s/old/new/' file.txt

# Delete lines
sed '2d' file.txt

# Delete range of lines
sed '2,5d' file.txt

# Insert text
sed '2i\New line' file.txt

# In-place editing (modify file)
sed -i 's/old/new/g' file.txt
```

### `awk` - Pattern Scanning and Processing
```bash
# Print specific columns
awk '{print $1, $3}' file.txt

# Field separator
awk -F: '{print $1}' /etc/passwd

# Sum column values
awk '{sum+=$1} END {print sum}' file.txt

# Print lines matching pattern
awk '/pattern/ {print}' file.txt

# Conditional printing
awk '$3 > 100 {print $1, $3}' file.txt
```

### `sort` - Sort Lines
```bash
# Basic sort
sort file.txt

# Numeric sort
sort -n file.txt

# Reverse sort
sort -r file.txt

# Sort by specific column
sort -k2 file.txt

# Remove duplicates
sort -u file.txt
```

### `uniq` - Report or Filter Repeated Lines
```bash
# Remove duplicate adjacent lines
uniq file.txt

# Count occurrences
uniq -c file.txt

# Show only duplicate lines
uniq -d file.txt

# Show only unique lines
uniq -u file.txt
```

### `wc` - Word, Line, and Character Count
```bash
# All counts (lines, words, characters)
wc file.txt

# Line count
wc -l file.txt

# Word count
wc -w file.txt

# Character count
wc -c file.txt
```

### `tr` - Translate Characters
```bash
# Replace characters
echo "Hello" | tr 'a-z' 'A-Z'  # Output: HELLO

# Delete characters
echo "Hello 123" | tr -d '0-9'  # Output: Hello 

# Squeeze repeating characters
echo "Hello    World" | tr -s ' '  # Output: Hello World
```

### `cut` - Extract Sections from Files
```bash
# Cut by character position
cut -c1-5 file.txt

# Cut by delimiter and field
cut -d':' -f1 /etc/passwd

# Output delimiter
cut -d':' -f1,3 --output-delimiter=' ' /etc/passwd
```

### `paste` - Merge Lines of Files
```bash
# Merge files horizontally
paste file1.txt file2.txt

# Specify delimiter
paste -d':' file1.txt file2.txt
```

### `join` - Join Lines on Common Field
```bash
join file1.txt file2.txt
```

### `diff` - Compare Files
```bash
# Standard diff
diff file1.txt file2.txt

# Side-by-side comparison
diff -y file1.txt file2.txt

# Unified format
diff -u file1.txt file2.txt
```

## Finding Files and Content

### `find` - Search for Files
```bash
# Find by name
find /path -name "*.txt"

# Find by type
find /path -type f  # files
find /path -type d  # directories

# Find by size
find /path -size +10M  # larger than 10MB
find /path -size -1M   # smaller than 1MB

# Find by modification time
find /path -mtime -7  # modified in last 7 days

# Find and execute command
find /path -name "*.txt" -exec rm {} \;

# Find and print full path
find /path -name "*.txt" -print
```

### `locate` - Find Files by Name (uses database)
```bash
locate filename
```

### `updatedb` - Update locate Database
```bash
sudo updatedb
```

### `which` - Show Full Path of Commands
```bash
which python
```

### `whereis` - Locate Binary, Source, and Manual Files
```bash
whereis python
```

## Compression and Archives

### `tar` - Tape Archive
```bash
# Create archive
tar -cvf archive.tar directory/

# Create gzipped archive
tar -czvf archive.tar.gz directory/

# Create bzip2 archive
tar -cjvf archive.tar.bz2 directory/

# Extract archive
tar -xvf archive.tar

# Extract gzipped archive
tar -xzvf archive.tar.gz

# Extract bzip2 archive
tar -xjvf archive.tar.bz2

# List contents
tar -tvf archive.tar
```

### `gzip` - Compress Files
```bash
# Compress file
gzip file.txt  # creates file.txt.gz and removes original

# Decompress file
gzip -d file.txt.gz

# Keep original file
gzip -k file.txt
```

### `gunzip` - Decompress Files
```bash
gunzip file.txt.gz
```

### `bzip2` - Compress Files (better compression)
```bash
# Compress file
bzip2 file.txt

# Decompress file
bzip2 -d file.txt.bz2
```

### `zip` - Create Zip Archive
```bash
# Create zip archive
zip archive.zip file1.txt file2.txt

# Create zip archive from directory
zip -r archive.zip directory/
```

### `unzip` - Extract Zip Archive
```bash
# Extract zip archive
unzip archive.zip

# List contents
unzip -l archive.zip
```

## Networking Commands

### `ping` - Test Connectivity
```bash
ping google.com
```

### `traceroute` - Trace Route to Host
```bash
traceroute google.com
```

### `dig` - DNS Lookup
```bash
dig google.com
```

### `nslookup` - Query DNS Records
```bash
nslookup google.com
```

### `whois` - Domain Information
```bash
whois google.com
```

### `ifconfig` - Network Interface Configuration
```bash
ifconfig
```

### `ip` - Modern Network Configuration
```bash
# Show address info
ip addr

# Show routing table
ip route
```

### `netstat` - Network Statistics
```bash
# List all connections
netstat -a

# List TCP connections
netstat -t

# List listening ports
netstat -l

# Show process ID and program name
netstat -p
```

### `ss` - Socket Statistics (modern netstat)
```bash
ss -a
```

### `wget` - Download Files
```bash
# Download file
wget https://example.com/file.txt

# Download to specific location
wget -O output.txt https://example.com/file.txt

# Download recursively
wget -r https://example.com/
```

### `curl` - Transfer Data from/to Server
```bash
# Get web page
curl https://example.com

# Download file
curl -o output.txt https://example.com/file.txt

# Send POST request
curl -X POST -d "data" https://example.com

# Send with headers
curl -H "Content-Type: application/json" https://example.com
```

### `ssh` - Secure Shell
```bash
# Connect to remote host
ssh user@hostname

# Specify port
ssh -p 2222 user@hostname

# Use identity file
ssh -i key.pem user@hostname
```

### `scp` - Secure Copy
```bash
# Copy file to remote host
scp file.txt user@hostname:/path/

# Copy file from remote host
scp user@hostname:/path/file.txt local_directory/

# Copy recursively
scp -r directory/ user@hostname:/path/
```

## Disk Usage

### `du` - Disk Usage
```bash
# Directory size summary
du -sh directory/

# All subdirectories
du -h directory/

# Sort by size
du -h | sort -hr
```

### `df` - Disk Free Space
```bash
# All filesystems
df -h

# Specific filesystem
df -h /dev/sda1
```

### `ncdu` - NCurses Disk Usage
```bash
ncdu /path
```

## SSH and Remote Access

### `ssh-keygen` - Generate SSH Keys
```bash
# Generate key pair
ssh-keygen -t rsa -b 4096

# Specify file
ssh-keygen -t rsa -b 4096 -f ~/.ssh/mykey
```

### `ssh-copy-id` - Install SSH Key
```bash
ssh-copy-id user@hostname
```

### `rsync` - Remote File Synchronization
```bash
# Sync local to remote
rsync -avz source/ user@hostname:/destination/

# Sync remote to local
rsync -avz user@hostname:/source/ destination/

# Dry run (no changes made)
rsync -avz --dry-run source/ destination/

# Delete files in destination not in source
rsync -avz --delete source/ destination/
```

## Shell Scripting Basics

### Shebang Line
```bash
#!/bin/bash
```

### Variables
```bash
# Assign variable
NAME="John"

# Access variable
echo $NAME

# Command output to variable
DATE=$(date)
echo $DATE
```

### Conditionals
```bash
if [ "$NAME" = "John" ]; then
    echo "Hello John"
elif [ "$NAME" = "Jane" ]; then
    echo "Hello Jane"
else
    echo "Hello stranger"
fi
```

### Loops
```bash
# For loop
for i in 1 2 3 4 5; do
    echo $i
done

# While loop
COUNT=1
while [ $COUNT -le 5 ]; do
    echo $COUNT
    COUNT=$((COUNT+1))
done

# Until loop
COUNT=1
until [ $COUNT -gt 5 ]; do
    echo $COUNT
    COUNT=$((COUNT+1))
done
```

### Functions
```bash
# Define function
greet() {
    echo "Hello, $1!"
}

# Call function
greet "World"
```

### Arguments
```bash
echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"
```

### Exit Status
```bash
# Get last command exit status
echo $?

# Set exit status
exit 0  # Success
exit 1  # Failure
```

## Bioinformatics Tools

### Sequence Analysis

#### `fastqc` - Quality Control for Sequencing Data
```bash
# Basic usage
fastqc sequence.fastq

# Multiple files
fastqc sample1.fastq sample2.fastq

# Output directory
fastqc -o output_dir/ sample.fastq

# Extract results
fastqc -o output_dir/ --extract sample.fastq
```

#### `cutadapt` - Trim Adapter Sequences
```bash
# Trim adapter
cutadapt -a ADAPTER -o output.fastq input.fastq

# Trim paired-end
cutadapt -a ADAPTER1 -A ADAPTER2 -o out1.fastq -p out2.fastq in1.fastq in2.fastq

# Quality trimming
cutadapt -q 20 -o output.fastq input.fastq
```

#### `trimmomatic` - Flexible Read Trimming
```bash
# Single-end
trimmomatic SE input.fastq output.fastq ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36

# Paired-end
trimmomatic PE -phred33 input_1.fastq input_2.fastq output_1_paired.fastq output_1_unpaired.fastq output_2_paired.fastq output_2_unpaired.fastq ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
```

#### `seqtk` - Sequence Toolkit
```bash
# Convert FASTQ to FASTA
seqtk seq -a input.fastq > output.fasta

# Extract sequences
seqtk subseq input.fasta list.txt > output.fasta

# Sample sequences
seqtk sample input.fastq 10000 > sample.fastq
```

#### `samtools` - SAM/BAM/CRAM Manipulation
```bash
# Convert SAM to BAM
samtools view -b -o output.bam input.sam

# Sort BAM file
samtools sort -o sorted.bam input.bam

# Index BAM file
samtools index sorted.bam

# View BAM file
samtools view sorted.bam

# View specific region
samtools view sorted.bam chr1:10000-20000

# Extract FASTA from reference
samtools faidx reference.fasta chr1:1000-2000

# Calculate coverage
samtools depth sorted.bam > coverage.txt

# Get statistics
samtools flagstat sorted.bam
```

#### `bedtools` - Toolset for Genome Arithmetic
```bash
# Intersect BED files
bedtools intersect -a file1.bed -b file2.bed > intersection.bed

# Get FASTA from BED regions
bedtools getfasta -fi reference.fasta -bed regions.bed -fo output.fasta

# Genome coverage
bedtools genomecov -ibam sorted.bam -g genome.file > coverage.bed

# Merge overlapping features
bedtools merge -i sorted.bed > merged.bed

# Find complement (not in BED)
bedtools complement -i features.bed -g genome.file > complement.bed
```

### Alignment Tools

#### `bwa` - Burrows-Wheeler Aligner
```bash
# Index reference
bwa index reference.fasta

# Alignment (original)
bwa aln reference.fasta reads.fastq > aln.sai
bwa samse reference.fasta aln.sai reads.fastq > aligned.sam

# Alignment (mem algorithm)
bwa mem reference.fasta reads.fastq > aligned.sam

# Paired-end alignment
bwa mem reference.fasta read1.fastq read2.fastq > aligned.sam
```

#### `bowtie2` - Fast and Sensitive Alignment
```bash
# Build index
bowtie2-build reference.fasta reference

# Single-end alignment
bowtie2 -x reference -U reads.fastq -S aligned.sam

# Paired-end alignment
bowtie2 -x reference -1 reads1.fastq -2 reads2.fastq -S aligned.sam

# Local alignment
bowtie2 --local -x reference -U reads.fastq -S aligned.sam
```

#### `hisat2` - Hierarchical Indexing for Spliced Alignment
```bash
# Build index
hisat2-build reference.fasta reference

# Single-end alignment
hisat2 -x reference -U reads.fastq -S aligned.sam

# Paired-end alignment
hisat2 -x reference -1 reads1.fastq -2 reads2.fastq -S aligned.sam
```

#### `star` - Spliced Transcripts Alignment to Reference
```bash
# Generate genome index
STAR --runMode genomeGenerate --genomeDir index/ --genomeFastaFiles reference.fasta --sjdbGTFfile annotation.gtf

# Single-end alignment
STAR --genomeDir index/ --readFilesIn reads.fastq --outFileNamePrefix output

# Paired-end alignment
STAR --genomeDir index/ --readFilesIn reads1.fastq reads2.fastq --outFileNamePrefix output
```

#### `minimap2` - Versatile Sequence Aligner
```bash
# Map long reads to reference
minimap2 -a reference.fasta long_reads.fastq > aligned.sam

# Map short reads
minimap2 -ax sr reference.fasta reads.fastq > aligned.sam

# Splice-aware alignment
minimap2 -ax splice reference.fasta reads.fastq > aligned.sam
```

### Assembly Tools

#### `spades` - De Novo Genome Assembler
```bash
# Basic assembly
spades.py -o output_dir -1 reads1.fastq -2 reads2.fastq

# With long reads
spades.py -o output_dir -1 reads1.fastq -2 reads2.fastq --pacbio pb_reads.fastq

# RNA assembly
rnaspades.py -o output_dir -1 reads1.fastq -2 reads2.fastq
```

#### `canu` - Long Read Assembler
```bash
canu -p prefix -d output_dir genomeSize=5m -pacbio-raw reads.fastq
```

#### `velvet` - Short Read Assembler
```bash
# Build hash table
velveth output_dir 31 -fastq reads.fastq

# Assemble contigs
velvetg output_dir -cov_cutoff auto
```

#### `megahit` - Ultra-fast NGS Assembler
```bash
# Single-end
megahit -r reads.fastq -o output_dir

# Paired-end
megahit -1 reads1.fastq -2 reads2.fastq -o output_dir
```

### Variant Calling

#### `bcftools` - Utilities for Variant Calling
```bash
# Call variants
bcftools mpileup -f reference.fasta sorted.bam | bcftools call -mv -o variants.vcf

# Filter variants
bcftools filter -i 'QUAL>20 && DP>10' variants.vcf > filtered.vcf

# Convert to BCF
bcftools view -O b -o variants.bcf variants.vcf

# Stats
bcftools stats variants.vcf > stats.txt
```

#### `gatk` - Genome Analysis Toolkit
```bash
# Mark duplicates
gatk MarkDuplicates -I input.bam -O marked_duplicates.bam -M metrics.txt

# Base recalibration
gatk BaseRecalibrator -I input.bam -R reference.fasta --known-sites known_sites.vcf -O recal_data.table

# Apply BQSR
gatk ApplyBQSR -I input.bam -R reference.fasta --bqsr-recal-file recal_data.table -O recalibrated.bam

# Call variants
gatk HaplotypeCaller -R reference.fasta -I input.bam -O output.vcf

# Joint genotyping
gatk GenotypeGVCFs -R reference.fasta -V input.g.vcf -O output.vcf
```

#### `freebayes` - Bayesian Variant Calling
```bash
freebayes -f reference.fasta input.bam > variants.vcf
```

#### `snpEff` - Variant Annotation
```bash
# Annotate variants
snpEff ann -v GRCh38.86 input.vcf > annotated.vcf

# Build database
snpEff build -v GRCh38.86
```

### RNA-Seq Analysis

#### `featureCounts` - Read Counting for RNA-Seq
```bash
featureCounts -a annotation.gtf -o counts.txt input.bam
```

#### `htseq-count` - Count Reads in Features
```bash
htseq-count -f bam -r pos -s no -t exon -i gene_id aligned.bam annotation.gtf > counts.txt
```

#### `salmon` - Transcript Quantification
```bash
# Index
salmon index -t transcripts.fasta -i salmon_index

# Quantify
salmon quant -i salmon_index -l A -1 reads1.fastq -2 reads2.fastq -o salmon_output
```

#### `kallisto` - Fast Transcript Quantification
```bash
# Index
kallisto index -i index transcripts.fasta

# Quantify
kallisto quant -i index -o output -b 100 reads1.fastq reads2.fastq
```

#### `stringtie` - Transcript Assembly and Quantification
```bash
# Assemble transcripts
stringtie aligned.bam -G annotation.gtf -o assembled.gtf

# Quantify with reference
stringtie aligned.bam -G annotation.gtf -e -o output.gtf -A gene_abund.tab
```

### Metagenomics

#### `kraken2` - Taxonomic Classification
```bash
# Build database
kraken2-build --standard --db kraken_db

# Classify reads
kraken2 --db kraken_db --output output.txt --report report.txt input.fastq
```

#### `metaphlan` - Profiling Composition
```bash
metaphlan input.fastq --input_type fastq --bowtie2out bowtie2.bz2 --nproc 4 --output_file profiled_metagenome.txt
```

#### `humann` - Functional Profiling
```bash
humann --input input.fastq --output humann_output
```

### Visualization

#### `igv` - Integrative Genomics Viewer
```bash
# Start IGV
igv

# Batch script
igv -b batch_script.txt
```

#### `circos` - Circular Genome Visualization
```bash
circos -conf circos.conf
```

#### `tablet` - Next Generation Sequence Assembly Viewer
```bash
tablet
```