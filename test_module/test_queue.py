from queue import Queue
import numpy

q = Queue()

# print(type(x))
def at():
    x = 'toan'
    try:
        x = numpy.int64(x)
        # print(type(x))
        # print('Toan')
    except Exception as e:
        # print(e)
        raise Exception("toan")
try:
    at()
    # print('vao')
except Exception as e:
    print('tu ham nho:', e)
    # Đưa đối tượng Exception vào hàng đợi
    q.put(e)
    
while not q.empty():
    item = q.get()
    if isinstance(item, Exception):
        print('phai')
    else:
        print('khong phai')
