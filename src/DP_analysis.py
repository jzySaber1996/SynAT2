from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree

nlp = StanfordCoreNLP('E:\\stanford-corenlp-latest\\stanford-corenlp-4.4.0')

s = 'CSRF attacks'

# print ('Tokenize:', nlp.word_tokenize(s))
# print ('Part of Speech:', nlp.pos_tag(s))
# print ('Named Entities:', nlp.ner(s))
print('Constituency Parsing:', nlp.parse(s))#语法树
# print ('Dependency Parsing:', nlp.dependency_parse(s))#依存句法

tree=Tree.fromstring(nlp.parse(s))
# tree.draw()

nlp.close()