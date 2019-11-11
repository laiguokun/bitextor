from toolwrapper import ToolWrapper
from func_timeout import func_timeout, FunctionTimedOut
from external_processor import ExternalTextProcessor
import string
import sys
import html


def filter_digits_and_punctuation(original_text):
    text_split = original_text.split()
    if len(text_split) == 1 and sum([1 for m in text_split[0] if m in string.punctuation + string.digits]) > len(
            text_split[0]) // 2:
        return False
    return True


def check_buffering(command):
    proc = ToolWrapper(command)
    proc.writeline('hello world\n')
    try:
        func_timeout(10, proc.readline)
    except FunctionTimedOut:
        sys.stderr.write("Command '{}' is buffering\n".format(" ".join(command)))
        return True
    return False


class Tokenizer(object):
    def __init__(self, command):
        buffering = check_buffering(command)
        if buffering:
            self.proc = ExternalTextProcessor(command)
            self.subprocess = True
        else:
            self.proc = ToolWrapper(command)
            self.subprocess = False

    def tokenize(self, segment):
        if self.subprocess:
            tokenized_text = self.proc.process(segment)
        else:
            self.proc.writeline(segment)
            tokenized_text = self.proc.readline()
        return tokenized_text.strip()


class SentenceSplitter(object):

    def __init__(self, cmd):
        buffering = check_buffering(cmd)
        if buffering:
            self.proc = ExternalTextProcessor(cmd)
            self.subprocess = True
        else:
            self.proc = ToolWrapper(cmd)
            self.subprocess = False
            self.delimiter = self.get_delimiter()
            if not self.delimiter.strip():
                sys.stderr.write("No delimiter for {}, using subprocess\n".format(cmd))
                self.subprocess = True
                self.proc = ExternalTextProcessor(cmd)

    def split_sentences(self, content, delimiter="<P>"):
        content = html.escape(content)
        if not delimiter.strip():
            delimiter = self.delimiter
        segments = ""
        if self.subprocess:
            segments = self.proc.process(content)
        else:
            segment = ""
            self.proc.writeline(content.strip() + "\n")
            while segment != delimiter:
                segment = self.proc.readline()
                if segment != delimiter and segment != "":
                    segments = segments + segment.strip() + "\n"
        return html.unescape(segments).strip()

    def get_delimiter(self):
        if self.subprocess:
            return ""
        self.proc.writeline("hello\n")
        try:
            func_timeout(10, self.proc.readline)
            delimiter = func_timeout(10, self.proc.readline)
            return delimiter
        except FunctionTimedOut:
            return ""


