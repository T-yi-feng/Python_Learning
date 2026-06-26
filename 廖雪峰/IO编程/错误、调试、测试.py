#--------------------错误、调试、测试----------------------
# #1、错误处理
# def try_1(*args, **kwargs):
#     try:
#         print('try...')
#         r = 10 / int(*args[0])
#         print('result:', r)
#     except ValueError as e:
#         print('ValueError:', e)
#     except ZeroDivisionError as e:
#         print('ZeroDivisionError:', e)
#     else:
#         print('no error!')
#     finally:
#         print('finally...')
#     print('END')


# try_1('2')
# try_1('0')
        
# 使用try...except捕获错误还有一个巨大的好处,就是可以跨越多层调用：
# 比如函数main()调用bar()，bar()调用foo()，结果foo()出错了。
#     这时，只要main()捕获到了，就可以处理：
# def foo(s):
#     return 10/int(s)

# def bar(s):
#     return foo(s)*2

# def main():
#     try:
#         bar('0')
#     except Exception as e:
#         print('Error:', e)
#     finally:
#         print('finally...')

# main()

# 也就是说，不需要在每个可能出错的地方去捕获错误，
# 只要在合适的层次去捕获错误就可以了。
# 这样一来，就大大减少了写try...except...finally的麻烦。



# 记录错误：
# 如果不捕获错误，自然可以让Python解释器来打印出错误堆栈，但程序也被结束了。
# 既然我们能捕获错误，就可以把错误堆栈打印出来，然后分析错误原因，
# 同时，让程序继续执行下去。

# import logging

# def foo(s):
#     return 10/int(s)

# def bar(s):
#     return foo(s)*2

# def main():
#     try:
#         bar('0')
#     except Exception as e:
#         logging.exception(e)

# main()
# print('END')

#虽然运行会出错，但是程序并没有被结束，而是继续执行下去，并且错误堆栈被打印出来了。
#直到输出END，说明程序已经继续执行了。


# 抛出错误raise error：
# class FooError(ValueError):
#     pass
# def foo(s):
#     n = int(s)
#     if n == 0:
#         raise FooError('invalid value: %s' % s)
#     return 10 / n

# foo('0')



#2、调试
#调试方法1：使用print()函数：
#调试方法2：使用断言assert：(能用print的就一定可以用断言assert)
# def foo(s):
#     n = int(s)
#     assert n != 0, 'n is zero!'
#     #assert的意思是:
#     #   现在断言n不为零；
#     #   如果n为零，就抛出AssertionError，并且错误消息为'n is zero!'
#     return 10 / n

# def main():
#     foo('0')

# main()

#调试方法3：使用logging模块：

# import logging
# #它的好处是：1.可以指定记录信息的级别，允许你过滤掉不需要的信息，
# #           同时它还会把错误堆栈信息一并输出.
# #           2.logging可以记录到文件里，例如console，方便事后排查。 
# logging.basicConfig(level=logging.ERROR)
# #这就是logging的好处，它允许你指定记录信息的级别，
# #   有debug，info，warning，error等几个级别
# s = '0'
# n = int(s)
# logging.info('n = %d' % n)
# print(10 / n)


#调试方法4：使用pdb模块：
#第4种方式是启动Python的调试器pdb:
#   让程序以单步方式运行，可以随时查看运行状态。(不咋用，感觉不如print和logging方便)

# import pdb 

# s = '0'
# n = int(s)
# pdb.set_trace()
# print(10 / n)

#调试方法5：使用IDE：
#IDE（集成开发环境）是专门为程序员设计的编辑器，



#3、单元测试✨
# 这种以测试为驱动的开发模式最大的好处就是：
# 确保一个程序模块的行为符合我们设计的测试用例，
# 在将来修改的时候，可以极大程度地保证该模块行为仍然是正确的。

#编写一个类，让类的行为与dict一致：
#mydict.py编写如下：
class Dict(dict):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def __getattr__(self,key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)
    
    def __setattr__(self,key,value):
        self[key] = value

# #为了编写单元测试，我们需要引入Python自带的unittest模块。
# #   编写如下：
import unittest
# #编写单元测试时，我们需要编写一个测试类，从unittest.TestCase继承。
# #每个test_xxx()方法就是一个测试用例，测试方法的名字必须以test开头，才能被自动执行。

class TestDict(unittest.TestCase):
    def test_init(self):
        d = Dict(a=1,b='test')
        self.assertEqual(d.a,1)             #断言d.a的值是否为1
        self.assertEqual(d.b,'test')        #断言d.b的值是否为'test'
        self.assertTrue(isinstance(d,dict)) #断言d是否是dict的实例
#         #assertEqual()是测试断言，如果断言与实际不符，会报错
    
    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key,'value')     #断言d.key的值是否为'value'

    def test_attr(self):
        d = Dict()
        d.key = 'value'
        self.assertTrue(hasattr(d,'key'))   #断言d是否有属性'key'
        self.assertEqual(d['key'],'value')  #断言d['key']的值是否为'value'

    def test_keyerror(self):
        d = Dict()
        with self.assertRaises(KeyError):    
#             #断言访问不存在的key时是否会抛出KeyError
            value = d['empty']

    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
#             #断言访问不存在的属性时是否会抛出AttributeError
            value = d.empty

# #运行单元测试：
if __name__ == '__main__':
    unittest.main()

#SetUp和TearDown：
#如果我们编写了很多测试用例，并且每个测试用例都需要执行一些相同的代码，
#    例如创建一个Dict()对象，那么我们就可以把这些代码放到setUp()方法中，
#    这样每个测试用例在运行前都会调用setUp()方法，

# class TestDict(unittest.TestCase):
#     def setUp(self):
#         self.d = Dict(a=1,b='test')

#     def test_init(self):
#         self.assertEqual(self.d.a,1)
#         self.assertEqual(self.d.b,'test')
#         self.assertTrue(isinstance(self.d,dict))

#     def test_key(self):
#         self.d['key'] = 'value'
#         self.assertEqual(self.d.key,'value')

#     def test_attr(self):
#         self.d.key = 'value'
#         self.assertTrue(hasattr(self.d,'key'))
#         self.assertEqual(self.d['key'],'value')

#     def test_keyerror(self):
#         with self.assertRaises(KeyError):
#             value = self.d['empty']

#     def test_attrerror(self):
#         with self.assertRaises(AttributeError):
#             value = self.d.empty
    
#     def tearDown(self):
#         pass
#如果setUp()方法中创建了数据库连接，那么tearDown()方法就可以把数据库连接关闭，
#    这样无论测试成功还是失败，都能保证连接正确地关闭了。
# if __name__ == '__main__':
#     unittest.main()


#4、文档测试



















