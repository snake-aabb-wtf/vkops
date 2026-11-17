import chat,ops

def main():
 gpu=ops.GPU();eng=object.__new__(chat.Qwen3Engine)
 eng.gpu=gpu;eng.H=2560;eng.inter=9728;eng.num_layers=1
 eng._alloc(2)
 needed=2*chat.HEADS*chat.HEAD_DIM*2
 assert eng.s['att16'].size>=needed,(eng.s['att16'].size,needed)
 print('PASS full-capacity attention projection buffer',needed)
if __name__=='__main__':main()
