#!/usr/bin/env python3
import requests

######################################################################################
def ConvertEncoding(data, encoding):
    if encoding is not None and len(data) > 0:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            sys.stderr.write("encoding error")
    return ''

######################################################################################

url = "http://www.buchmann.ch"
#url = "https://www.buchmann.ch/catalog/default.php"
pageResponse = requests.get(url, timeout=5)
print("status_code", pageResponse.status_code)

for histResponse in pageResponse.history:
    print("   histResponse", histResponse, histResponse.url, \
            histResponse.apparent_encoding, histResponse.encoding)
    print(histResponse.text)

print("pageResponse", pageResponse, pageResponse.url, \
        pageResponse.apparent_encoding, pageResponse.encoding)
#print(pageResponse.text)

text = pageResponse.text
#text = ConvertEncoding(pageResponse.text, pageResponse.encoding)

content = pageResponse._content

with open("text", "w") as f:
    f.write(text)

with open("content", "wb") as f:
    f.write(content)
