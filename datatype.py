#
import codecs
import _pickle as cPickle


path1 = './feature_2classify_3/train/1_record425702358_22651761.p'
text =cPickle.load(open(path1, 'rb'))

txtfile = './txt/1_record425702358_22651761.txt'

print(text.shape)
print(text[20][0])
print(text[20][62])

with open(txtfile, "w") as f:
    for i in range(0, text.shape[0]):
        for j in range(text.shape[1]):
            f.write(str(text[i, j]) + ' ')
        f.write('\r\n')



# f = codecs.open(txtfile, 'w', 'utf-8')
# for i in range(0, text.shape[0]):
#     for j in range(text.shape[1]):
#         f.write(str(text[i, j]) + ',')
#     f.write('\r\n')
# f.close()
