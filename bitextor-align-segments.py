#!/usr/bin/env python3


#  This file is part of Bitextor.
#
#  Bitextor is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Bitextor is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Bitextor.  If not, see <https://www.gnu.org/licenses/>.


# 1. Reading from STDIN a set of aligned documents. The input format is:
#   filename1	filename2	clean_text1_in_base64	clean_text2_in_base64
# 2. Text is cleaned and, for every aligned pair, both texts are dumped, in the same order in two temporary files.
# Every text block is sepparated to the previous one by a block:
#    <p>
#    <file lang="lang_id">file_name</file>
#    <p>
# 3. Running hunalign on the two temporary files
# 4. Removing unaligned segments and <p> mark
# 5. Identifying the filenames for every block of segments, and printing everything to the output
#
# Output format:
#   filename1    filename2    segment1    segment2    quality
#

import sys
import os
import argparse
import base64
import subprocess
from tempfile import NamedTemporaryFile


def run_aligner(filename_s, filename_t, dic, hunaligndir):
    # option -ppthresh=10?
    if dic is None or dic == "":
        if hunaligndir is None:
            hunalign = [os.path.dirname(os.path.abspath(__file__)) + "hunalign", "-realign", "/dev/null", filename_s,
                        filename_t]
        else:
            hunalign = [hunaligndir + "/hunalign", "-realign", "/dev/null", filename_s, filename_t]
    else:
        if hunaligndir is None:
            hunalign = [os.path.dirname(os.path.abspath(__file__)) + "hunalign", dic, filename_s, filename_t]
        else:
            hunalign = [hunaligndir + "/hunalign", dic, filename_s, filename_t]

    p = subprocess.Popen(hunalign, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for line_o in p.stdout:
        yield line_o
    return

def align(file1, file2, file1orig, file2orig, dic):
    filereader1 = open(file1orig, "r")
    filereader2 = open(file2orig, "r")

    hunalign_output = run_aligner(file1, file2, dic, options.hunaligndir)
    try:
        prev_hun = next(hunalign_output).strip()
        prev_fields = prev_hun.split(b"\t")
        if int(prev_fields[0]) > 0:
            for i in range(int(prev_fields[0])):
                line1 = filereader1.readline().strip()

        if int(prev_fields[1]) > 0:
            for i in range(int(prev_fields[1])):
                line2 = filereader2.readline().strip()

    except StopIteration:
        prev_hun = ""
    for line_h in hunalign_output:
        hun_line = line_h.strip()
        last_position1 = filereader1.tell()
        last_position2 = filereader2.tell()
        line1 = filereader1.readline().strip()
        line2 = filereader2.readline().strip()
        prev_fields = prev_hun.split(b"\t")
        hunalign_fields = hun_line.split(b"\t")

        if float(prev_fields[2]) == -0.3:
            if int(hunalign_fields[0]) == int(prev_fields[0]):
                line1 = ""
                filereader1.seek(last_position1)
            elif int(hunalign_fields[1]) == int(prev_fields[1]):
                line2 = ""
                filereader2.seek(last_position2)

        if int(hunalign_fields[0]) - int(prev_fields[0]) > 1:
            for i in range((int(hunalign_fields[0]) - int(prev_fields[0])) - 1):
                line1 += " " + filereader1.readline().strip()

        if int(hunalign_fields[1]) - int(prev_fields[1]) > 1:
            for i in range((int(hunalign_fields[1]) - int(prev_fields[1])) - 1):
                line2 += " " + filereader2.readline().strip()

        print("{0}\t{1}\t{2}\t{3}\t{4}".format(filename1, filename2, line1, line2, prev_fields[2].decode("utf8")))

        prev_hun = hun_line

    filereader1.close()
    filereader2.close()


oparser = argparse.ArgumentParser(
    description="Tool that reads the output of bitextor-align-documents and aligns the segments of the aligned "
                "documents")
oparser.add_argument('aligned_docs', metavar='FILE', nargs='?',
                     help='File containing the set of aliged documents provided by the script '
                          'bitextor-align-documents (if undefined, the script reads from the standard input)',
                     default=None)
oparser.add_argument("--hunalign-dir",
                     help="Path to the installation of hunalign (for example: '/usr/local/bin'. If this option is not "
                          "defined, the executable will be searched in the same directory where this scritp is "
                          "placed.",
                     dest="hunaligndir", required=False, default=None)
oparser.add_argument("-d", help="Bilingual dictionary used for aligning and scoring", dest="dic", required=False,
                     default=None)
oparser.add_argument("-t", "--tmp-dir",
                     help="Temporary directory to be used for internal temporary files (/tmp by default)",
                     dest="tmpdir", required=False, default="/tmp")

options = oparser.parse_args()

if options.aligned_docs is None:
    reader_list = sys.stdin
else:
    reader_list = open(options.aligned_docs, "r")

for line in reader_list:
    tmp_file1 = NamedTemporaryFile(delete=False, dir=options.tmpdir)
    tmp_file2 = NamedTemporaryFile(delete=False, dir=options.tmpdir)
    tmp_file1_origtext = NamedTemporaryFile(delete=False, dir=options.tmpdir)
    tmp_file2_origtext = NamedTemporaryFile(delete=False, dir=options.tmpdir)

    fields = line.split("\t")
    filename1 = fields[0]
    filename2 = fields[1]
    encodedtext1 = fields[2]
    encodedtext2 = fields[3]
    encodedtokenized1 = fields[4]
    encodedtokenized2 = fields[5]

    tmp_file1_origtext.write(base64.b64decode(encodedtext1))
    tmp_file2_origtext.write(base64.b64decode(encodedtext2))

    tmp_file1.write(base64.b64decode(encodedtokenized1))
    tmp_file2.write(base64.b64decode(encodedtokenized2))

    tmp_file1_name = tmp_file1.name
    tmp_file2_name = tmp_file2.name
    tmp_file1_orig_name = tmp_file1_origtext.name
    tmp_file2_orig_name = tmp_file2_origtext.name

    tmp_file1.close()
    tmp_file1_origtext.close()
    tmp_file2.close()
    tmp_file2_origtext.close()

    align(tmp_file1_name, tmp_file2_name, tmp_file1_orig_name, tmp_file2_orig_name, options.dic)

    os.remove(tmp_file1.name)
    os.remove(tmp_file1_origtext.name)
    os.remove(tmp_file2.name)
    os.remove(tmp_file2_origtext.name)

