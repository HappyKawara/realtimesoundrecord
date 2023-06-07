class main:
    def __init__(self):
        self.a = 0
        self.b = 1
    def callback(self,x,y):
        print(x,y)
        print(self.a,self.b)
    def lll(self,func):
        return 10,11

l = main.callback
main.lll(main,l)
