#!/usr/bin/env python3
from enum import Enum, unique

@unique
class Gender(Enum):
    Male = 0
    Female = 1

class Student(object):
    def __init__(self, name,gender):
        self._name = name
        self._gender = gender
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, gender):
        if not isinstance(gender, Gender):
            raise ValueError('gender must be a Gender enum value')
        self._gender = gender

bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')

import os 
os.mkdir(os.path.join(os.path.abspath('.'),'IO编程'))