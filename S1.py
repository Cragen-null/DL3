import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:#ndarrayインスタンス以外のデータを入れた場合はエラーを出すようにする。
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None #微分した値(grad）を持たせるインスタンス変数。
        #ベクトルや行列など、多変数に関する微分は勾配(gradient）と呼ぶ。
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        #f = self.creator #1.関数を取得
        #if f is not None:
        #    x = f.input #2.関数の入力を取得
        #    x.grad = f.backward(self.grad) #3.関数のbackwardメソッドを呼ぶ。
        #    x.backward() #自分より一つ前の変数のbackwardメソッドを呼ぶ。（再起）
        if self.grad is None: #y.grad = np.array(1.0)を省略できるようにする。
            self.grad = np.ones_like(self.data)#self.dataと同じ形状かつ同じデータ型で、その要素が１のndarrayインスタンスを生成。
            
        #ループを用いた実装
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #関数を取得
            x, y = f.input, f.output # 関数の入出力を取得
            x.grad = f.backward(y.grad) #backwardメソッドを呼ぶ。
            
            if x.creator is not None:
                funcs.append(x.creator) #一つ前の関数をリストに追加


#ndarray関数か判定し、違うなら変換する。
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data #データを取り出す
        #y = x ** 2 #実際の計算 練習用
        y = self.forward(x) #具体的な計算はforwardメソッドで行う。
        output = Variable(as_array(y)) #Variableとして返す。
        output.set_creator(self) #出力変数に生みの親を覚えさせる。
        self.input = input #入力された変数を覚える。
        self.output = output#出力も覚える。
        return output
    
    def forward(self, x):
        raise NotImplementedError()
#あえて例外を発生させることで、Functionクラスのforwardメソッドを使った人（使ってしまった人）に対して、そのメソッドは継承して実装すべきであるとアピールする。
        
    def backward(self, gy):
        raise NotImplementedError()



class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

#微分を求める
def numerical_diff(f, x, eps=1e-4): #eps = epsilon
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


#合成関数の微分
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))





#array １次元配列
data = np.array(1.0)
print(data)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

#ndimインスタンス使用例 ndim=number of dimensionsの略で「次元の数」を表す。
y = np.array(1)
print(y.ndim)

y = np.array([1,2,3])
print(y.ndim)

y = np.array([[1,2,3],[4,5,6]])
print(y.ndim)

#Functionクラスを使う。
#__call__メソッドは fs = Function()としたとき、fs(...)と書くことで__call__メソッドを呼び出せる。

x = Variable(np.array(10))
#fs = Function() 練習用
fs = Square()
y = fs(x)


print(type(y)) #type()を使って、オブジェクトの形を取得
print(y.data)


#
#print(np.exp(1)) exp(0)==1.0 exp(1)==2.71828182845904
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
#print(b.data)
y = C(b)
print(y.data)

#numerical_diffを使う
#fs = Square()
#x = Variable(np.array(2.0))
#dy = numerical_diff(fs, x)
#print("-数値微分-")
#print(dy)


#合成関数の微分を使う
#x = Variable(np.array(0.5))
#dy = numerical_diff(f, x)
#print("-合成関数の微分-")
#print(dy)


#順伝番を求める。
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)
print(c.data)


#逆伝番によってyの微分を求める。
y.grad = np.array(1.0)
print(y.grad)
b.grad = C.backward(y.grad)
print(b.grad)
a.grad = B.backward(b.grad)
print(a.grad)
x.grad = A.backward(a.grad)
print("-yの逆伝番-")
print(x.grad)


#assert文を使てみる。assert ...　のように使い、ここで...がTrue出ない場合例外が発生する。
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)



#逆向きに計算グラフのノードを辿る
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

#逆伝番を試す。 x→A→a→B→ｂ→Ｃ→y
y.grad = np.array(1.0)

C = y.creator #1.関数を取得
b = C.input #2.関数の入力を取得
b.grad = C.backward(y.grad) #　関数のbackwardメソッドを呼ぶ。
B =b.creator #1.関数の取得
a = B.input #2.関数の入力を取得
a.grad = B.backward(b.grad) #3.関数のbackwardメソッドを呼ぶ。
A = a.creator #1.関数を取得
x = A.input #2.関数の入力を取得
x.grad = A.backward(a.grad) #3.関数のbackwardメソッドを呼ぶ。
print(x.grad)

#逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)


#square関数の実装
def square(x):
    return Square()(x)

#exp関数の実装
def exp(x):
    return Exp()(x)

#二つの関数を使ってみる
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)
#y = square(exp(square(x))) #連続して適用することもできる
y.grad = np.array(1.0)
y.backward()
print(x.grad)

#backwardメソッドの簡略化
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

#テストを行う
import unittest

class SquareTest(unittest.TestCase):
    #square関数の出力と「期待した値」が一致することを検証する。
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)#与えられた二つのオブジェクトが等しいか判定するメソッド。
        
    #square関数の逆伝播テスト
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    #勾配確認による自動テスト
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))#ランダムな入力値を生成。
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)#np.allclose(a, b)はndarrayインスタンスのaとbの値が近い値かどうか判定する。
        self.assertTrue(flg)
        
        

if __name__ == '__main__':
    unittest.main()

#import os
#os.path("D:\cragen\Programming\deep-learning-from-scratch-3-master")


#python -m unittest D:\cragen\Programming\deep-learning-from-scratch-3-master/steps/step10.py

