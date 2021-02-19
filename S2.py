import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True
    

class Variable:
    __array_priority__ = 200#Variableインスタンスの演算子の優先度を上げる。
    def __init__(self, data, name=None):
        if data is not None:#ndarrayインスタンス以外のデータを入れた場合はエラーを出すようにする。
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None #微分した値(grad）を持たせるインスタンス変数。
        #ベクトルや行列など、多変数に関する微分は勾配(gradient）と呼ぶ。
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad=False):
        #f = self.creator #1.関数を取得
        #if f is not None:
        #    x = f.input #2.関数の入力を取得
        #    x.grad = f.backward(self.grad) #3.関数のbackwardメソッドを呼ぶ。
        #    x.backward() #自分より一つ前の変数のbackwardメソッドを呼ぶ。（再起）
        if self.grad is None: #y.grad = np.array(1.0)を省略できるようにする。
            self.grad = np.ones_like(self.data)#self.dataと同じ形状かつ同じデータ型で、その要素が１のndarrayインスタンスを生成。
            
        #ループを用いた実装
        #funcs = [self.creator]
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop() #関数を取得
            #x, y = f.input, f.output # 関数の入出力を取得
            #x.grad = f.backward(y.grad) #backwardメソッドを呼ぶ。
            #gys = [output.grad for output in f.outputs]#出力変数であるoutputsの微分をリストにまとめる。
            gys = [output().grad for output in f.outputs]#弱参照用に書き換え。
            gxs = f.backward(*gys)#関数ｆの逆伝播を呼び出す。
            if not isinstance(gxs, tuple):#タプルでない場合、タプルへ変換。
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):#バックプロバゲーションで伝播する微分をＶａｒｉａｂｌｅインスタンス変数gradに設定する。
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    #funcs.append(x.creator) #一つ前の関数をリストに追加
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None#yはweakref
                    
                    
    def cleargrad(self):#微分を初期化するメソッド。
        self.grad = None
        
    @property#デコレーターを入れることでshapeメソッドはインスタンス変数としてアクセスできる。
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):#次元の数
        return self.data.ndim
    
    @property
    def size(self):#要素の数
        return self.data.size
    
    @property
    def dtype(self):#データの型
        return self.data.dtype
    
    #len関数をVariableでも利用できるようにする。（len関数はそこに含まれる要素の数を返す。）
    def __len__(self):
        return len(self.data)
    
    #print関数を使ってVariableの中身を確認できるようにする。
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)#ndarrayインスタンスを文字列に変換+__str__関数によって数値も文字列へ変換　
        return 'Variable(' + p + ')'
    
    #*演算子のオーバーロード
    def __mul__(self, other):
       return mul(self, other)

    
    


#ndarray関数か判定し、違うなら変換する。
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    #def __call__(self, input):
        #x = input.data #データを取り出す
        ##y = x ** 2 #実際の計算 練習用
        #y = self.forward(x) #具体的な計算はforwardメソッドで行う。
        #output = Variable(as_array(y)) #Variableとして返す。
        #output.set_creator(self) #出力変数に生みの親を覚えさせる。
        #self.input = input #入力された変数を覚える。
        #self.output = output#出力も覚える。
        #return output
    
    def __call__(self, *inputs):#*を付けることで、引数をまとめて受け取ることができる。
        inputs = [as_variable(x) for x in inputs]#各要素ｘをＶａｒｉａｂｌｅインスタンスへ変換
        xs = [x.data for x in inputs]  # Get data from Variable
        ys = self.forward(*xs)#*を付けてアンパッキング（リストを展開し渡す）。
        if not isinstance(ys, tuple):#タプルではない場合の追加対策。
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # Wrap data
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])#世代の設定
            for output in outputs:
                output.set_creator(self)#つながりの設定
            self.inputs = inputs
            #self.outputs = outputs#循環参照
            self.outputs = [weakref.ref(output) for output in outputs]#弱参照
        return outputs if len(outputs) > 1 else outputs[0]#リストの要素が1の時は最初の要素を返す。


    def forward(self, xs):
        raise NotImplementedError()
#あえて例外を発生させることで、Functionクラスのforwardメソッドを使った人（使ってしまった人）に対して、そのメソッドは継承して実装すべきであるとアピールする。
        
    def backward(self, gys):
        raise NotImplementedError()
        
class Add(Function):
    #def forward(self, xs):
    #    x0, x1 = xs
    #    y = x0 + x1
    #    return (y,)#タプルを返すようにする。
    #改善後
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
    
#AddクラスをPythonの関数として利用できるようにする。
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

#Multiplyクラス
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
#関数として使えるようにする。
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


#負数関数クラスを実装する。
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
#関数として利用できるようにする。    
def neg(x):
    return Neg()(x)

#Variableクラスに対して演算子のオーバーロードを行う。
Variable.__neg__ = neg


#引数
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


Variable.__sub__ = sub
Variable.__rsub__ = rsub


#割り算
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 **2)
        return gx0, gx1
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv


#累乗　底のｘについてのみ微分を求める。
class Pow(Function):
    def __init__(self, c):
        self.c = c
        
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow



class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        #x = self.input.data
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

#using_config関数
@contextlib.contextmanager#デコレーターを付けることでコンテキストを判断する関数が作られる。
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

#using_configをオフにするだけの関数。
def no_grad():
    return using_config('enable_backprop', False)



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


#square関数の実装
def square(x):
    return Square()(x)

#exp関数の実装
def exp(x):
    return Exp()(x)

#ndarrayインスタンスをVariableインスタンスに変換する関数
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

#+と*を扱う特殊メソッド
Variable.__mul__ = mul#演算子のオーバーロード
Variable.__add__ = add#演算子のオーバーロード
Variable.__rmul__ = mul
Variable.__radd__ = add




#addクラスの使い方
#xs = [Variable(np.array(2)), Variable(np.array(3))]#リストとして準備
#f = Add()
#ys = f(xs)#ysはタプル
#y = ys[0]
#print(y.data)

#addクラス改善後
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)


#新しいadd関数とsquare関数を使う。
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

#同じ変数を繰り返し使う
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)

#ダミー関数を用意してそれをfuncsというリストに追加する。
generations = [2, 0, 1, 4, 2]
funcs = []

for g in generations:
    f = Function()#ダミー関数クラス
    f.generation = g
    funcs.append(f)
    
print([f.generation for f in funcs])

#上のリストから世代の一番大きい関数を取り出す
funcs.sort(key=lambda x: x.generation)#並び替え
print([f.generation for f in funcs])
f = funcs.pop()#最後尾を取り出す
print(f.generation)

#動作確認
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)

#循環参照を持たないDeZeroで動作確認
for i in range(10):
    x = Variable(np.random.randn(10000))#巨大なデータ
    y = square(square(square(x)))#複雑な計算をする
                                                                                                                                                                 
#微分の保持状態の確認
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)
print(x0.grad, x1.grad)

#モード切り替え動作確認
Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()
#Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))

#using_config関数を使ってみる
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

#shapeメソッドを使ってみる
x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
print(x.shape)#x.shape()ではなく、x.shapeでアクセスできる。

#lenメソッドを使ってみる
x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
print(len(x))

#
x = Variable(np.array([1, 2, 3]))
print(x)

x = Variable(None)
print(x)

x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
print(x)


#mul関数を使ってみる。
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = add(mul(a, b), c)

y.backward()

print(y)
print(a.grad)
print(b.grad)


#*演算子を使ってみる。
Variable.__mul__ = mul#演算子のオーバーロード
Variable.__add__ = add#演算子のオーバーロード
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c + np.array(3.0)

y.backward()

print(y)
print(a.grad)
print(b.grad)

#
x = Variable(np.array(2.0))
y = 3.0 * x + 1.0
print(y)
x = Variable(np.array([1.0]))
y = np.array([2.0]) + x
print(y)

#neg
x = Variable(np.array(2.0))
y = -x
print(y)

#引き算
x = Variable(np.array(2.0))
y1 = 2.0 - x
y2 = x - 1.0
print(y1)
print(y2)

#累乗
x = Variable(np.array(2.0))
y = x ** 3
print(y)





#import numpy as np
#from dezero.core_simple import Variable
#from dezero import Variable

x = Variable(np.array(1.0))
print(x)


        # Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)


#Sphere関数
def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)


#matyas関数
def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    #演算子が使えない場合
    # z = sub(mul(0.26,add(pow(x, 2), pow(y, 2))),mul(0.48, mul(x, y)))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)


#Goldstein-Price関数
def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))* \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)
