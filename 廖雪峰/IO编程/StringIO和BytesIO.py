#------------ StringIO和BytesIO ------------#
#------------ StringIO----------------------#
#很多时候数据读取并不一定是文件，
#也有可能是在内存中读写：StringIO就是在内存中读写str.

# from io import StringIO

# f = StringIO()  #创建一个StringIO对象，像文件一样操作它

# f.write('hello')  #写入str

# #print(f)   
# #输出结果：<_io.StringIO object at 0x000001E8B9C1F2B0>
# print(f.getvalue())  #获取写入后的str
# #输出结果：hello
# f.write(' ')
# f.write('python_IO')
# print(f.getvalue())  #获取写入后的str
#输出结果：hello python_IO
#这种写入是追加写入，之前的内容不会被覆盖掉。

#如果要读取StringIO，可以用str初始化StringIO：
#然后和读取文件的line一样读取StringIO：
# f1 = StringIO('Hello！\nPython！\nStringIO')
# while True:
#     line = f1.readline()  #读取一行:
#     if line == '':
#         break
#     print(line.strip())  #strip()去掉末尾的空字符
#输出结果：
# Hello！
# Python！
# StringIO

#------------ BytesIO----------------------#
#StringIO操作的只能是str，如果要操作二进制数据，就需要BytesIO。
# BytesIO就是在内存中读写bytes。

# from io import BytesIO

#用法有点类似，但是只能写入bytes：
# f = BytesIO()
# f.write('中文'.encode('utf-8'))  #写入bytes，注意要编码成utf-8的bytes
# print(f.getvalue())  #获取写入后的bytes
#result: b'\xe4\xb8\xad\xe6\x96\x87'
#这个是utf-8编码的bytes，前面有个b表示这是bytes。
#故BytesIO是针对读写二进制Bytes数据的。

#如果要读取BytesIO，可以用bytes初始化BytesIO：(和StringIO一样

# f1 = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87\nPython!\nBytesIO')
# print(f1.read())  #读取BytesIO的全部内容
#result: b'\xe4\xb8\xad\xe6\x96\x87\nPython!\nBytesIO'

# BytesIO对象既有read()也有getvalue():
# - read() 从当前文件指针位置读取并移动指针，连续调用会返回空bytes直到重置指针。
# - getvalue() 返回缓冲区中全部bytes，不改变文件指针位置。
# 一般调试或需要完整内容时用getvalue()更方便；
# 按流式读取则用read().












