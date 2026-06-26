#------------操作文件和目录------------#
#操作文件和目录的函数一部分放在os模块中，另一部分放在os.path模块中.
#os模块包含创建、删除、重命名和获取文件属性等函数.
#os.path模块包含获取文件路径、检查路径是否存在等函数.
#os的全称就是：operating system，操作系统的意思.

#------------获取系统信息------------#
import os 
print('系统名称：',os.name)
#输出结果：nt
#如果是posix，说明系统是Linux、Unix或Mac OS X.
# 如果是nt，说明系统是Windows.

# print('详细系统信息：',os.uname())  # 仅在Linux/Unix/Mac OS X上可用

#------------获取环境变量------------#
#操作环境变量的方法是os.environ这个变量，它是一个包含所有环境变量的dict。

# print('环境变量：',os.environ) #输出很长.

#获取某个环境变量的值：
# print('PATH环境变量：',os.environ.get('PATH'))  
#输出结果：PATH环境变量： C:\Program Files\Python39\Scripts\;C:\...


#------------操作文件和目录------------#
#操作文件和目录的函数一部分放在os模块中，另一部分放在os.path模块中.

#查看当前的目录绝对路径：
print('当前目录的绝对路径：',os.path.abspath('.'))
#'.' 参数：
# 表示当前目录，输出结果：当前目录的绝对路径


#在某个目录下创建一个新目录：
# 1、首先把新目录的完整路径表示出来:

#✨拼接路径最好不要直接拼字符串。
# 而是利用os.path.join()函数，这样可以正确处理不同操作系统的路径分隔符。
new_dir = os.path.join('C:/Users/21495/Desktop/python', 'testdir')
#                                        完整路径     +    新目录名
#输出结果：' C:/Users/21495/Desktop/python/testdir'
print('新目录完整路径：', new_dir)
# 2、然后创建一个目录:
os.makedirs(new_dir, exist_ok=True)
#(make directory)创建一个目录，成功后无返回值，失败会抛出OSError的错误.

#------------验证文件夹是否创建成功------------#
if os.path.isdir(new_dir):
	print('文件夹创建成功：', new_dir)
else:
	print('文件夹创建失败：', new_dir)

#3、删除目录:
os.rmdir(new_dir)  #删除一个目录，成功后无返回值，失败会抛出OSError的错误.

#------------验证删除是否成功------------#
if not os.path.exists(new_dir):
	print('文件夹删除成功：', new_dir)
else:
	print('文件夹删除失败：', new_dir)


#如果是拆分路径，os.path模块提供了很多函数可以拆分路径。
#比如os.path.split()可以把一个路径拆分成两部分：
# 后一部分总是最后级别的目录或文件名，前面一部分是路径。
print('os.path.split()可以把路径拆分成两部分：',os.path.split(new_dir))
#输出结果：('C:/Users/21495/Desktop/python', 'testdir')

#moreover：
#  os.path.splitext()可以把文件名和扩展名拆分开：

file_path = os.path.abspath('Untitled-1.py')
print('os.path.splitext()可以把文件名和扩展名拆分开：',os.path.splitext(file_path))

#这些合并、拆分路径的函数并不要求目录和文件要真实存在，它们只对字符串进行操作。

#✨要列出所有的.py文件，只需一行代码：
py_files = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1] == '.py']
print('当前目录下的.py文件：')
for i in py_files:
	print(i)
	





















