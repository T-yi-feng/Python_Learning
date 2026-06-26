#-----------序列化-----------#
# 例如一个dict对象：
d = dict(name='Bob', age=20, score=88)
# 可以随时修改变量，比如把name改成'Bill'。
# 但是一旦程序结束，变量所占用的内存就被操作系统全部回收。
# 如果没有把修改后的'Bill'存储到磁盘上，下次重新运行程序，变量又被初始化为'Bob'。

#序列化就是做这个事情：
# 把变量从内存中变成可存储或传输的过程称为序列化，反过来就是反序列化。
# Python提供了pickle模块来实现序列化。
import pickle

#1、将一个对象序列化并写入文件：
d = dict(name = 'Bob',age = '20',score = 88)

#英语时间：dump意思是：1.沮丧；2.丢弃；3.倾销；4.转储；5.转存。
#所以pickle.dumps()的意思就是：把对象转储到为bytes。

print('序列化结果:',pickle.dumps(d))
#序列化结果: b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x...

#pickle.dumps()方法把任意对象序列化成一个bytes.
# 然后，就可以把这个bytes写入文件。
# 
# 或者用另一个方法pickle.dump()直接把对象序列化后写入一个file-like Object：
f = open('dump.txt','wb')  #以二进制写模式打开文件
pickle.dump(d, f)
f.close()

#注意，这种方法是把二进制数据写入文件的
# suite的文本编辑器是无法直接打开这个文件的。
# 但是我们可以用pickle.load()方法从文件中读取对象：

f = open('dump.txt','rb')
d1 = pickle.load(f)
print('反序列化结果：', d1)
f.close()

#--------------json模块----------------#
#更加常用的序列化方法是json模块。
# json模块的序列化和反序列化方法分别是json.dumps()和json.loads()。
import json
d = dict(name='Bob', age=20, score=88)
print('json序列化结果：', json.dumps(d))
#json序列化结果： {"name": "Bob", "age": 20, "score": 88}
#json.dumps()方法返回一个str，内容就是标准的JSON。

#如果要反序列化为Python对象，用json.loads()方法：
json_path = r"C:\\Users\\21495\\Desktop\\python\\Python-100-Days\\Day21-30\\草稿.json"
with open(json_path, 'r', encoding='utf-8') as f:
	data = json.load(f)  
	# ✨直接将JSON反序列化为Python对象（dict或list）

print('从json文件反序列化结果：', data)
print('json反序列化结果的类型：', type(data))

#-----------------JSON进阶----------------#
# Python的dict对象可以直接序列化为JSON的对象（{}），
# 但是有些特殊的对象，比如datetime、类Class对象，无法直接序列化为JSON.
# 例如：
class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
    
s = Student('Bob', 20, 88)
try:
    print(json.dumps(s))
except TypeError as e:
    print('序列化Student对象失败:', e, '这是因为Student对象无法直接序列化为JSON.')
    print('要序列化Student对象，可以提供一个函数来告诉json模块如何将Student对象转换为可序列化的类型。')
finally:
    pass

# 是因为默认情况下，dumps()方法不知道如何将Student实例变为一个JSON的{}对象。
# 可选参数default就是把任意一个对象变成一个可序列为JSON的对象.
# ✨我们只需要为Student专门写一个转换函数，再把函数传进去即可：

def student2dict(std):
     return{
        'name':std.name,
        'age':std.age,
        'score':std.score
     }

try:
    print('序列化成功：',json.dumps(s, default=student2dict))
except TypeError as e:
    print('序列化Student对象失败:', e, '这是因为Student对象无法直接序列化为JSON.')
    print('要序列化Student对象，可以提供一个函数来告诉json模块如何将Student对象转换为可序列化的类型。')
finally:
    pass



# 不过，下次如果遇到一个Teacher类的实例，照样无法序列化为JSON。
# 我们可以偷个懒，把任意class的实例变为dict：
def class2dict(obj):
    return obj.__dict__

#因为通常class的实例都有一个__dict__属性，它就是一个dict，用来存储实例变量。
# 也有少数例外，比如定义了__slots__的class.
# 则变为：
class teacher(object):
    # __slots__ = ('name', 'age', 'subject')  #定义了__slots__，没有__dict__
    #加上了slots之后，teacher对象就没有__dict__属性了，所以无法用class2dict函数来序列化了。
    def __init__(self, name, age, subject):
        self.name = name
        self.age = age
        self.subject = subject

s1 = teacher('Alice', 30, 'Math')

try:
    print('利用__dict__属性序列化成功：',json.dumps(s1, default=class2dict))
except TypeError as e:
    print('序列化Teacher对象失败:', e, '这是因为Teacher对象无法直接序列化为JSON.')
    print('要序列化Teacher对象，可以提供一个函数来告诉json模块如何将Teacher对象转换为可序列化的类型。')
finally:
    pass


