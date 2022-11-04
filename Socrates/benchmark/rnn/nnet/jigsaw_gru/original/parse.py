import numpy as np
import ast

input = open('model.txt', 'r')
lines = input.readlines()

print(len(lines))

for i in range(3):
    gwline = 1 + 4 * i
    gbline = 2 + 4 * i
    cwline = 3 + 4 * i
    cbline = 4 + 4 * i

    gw = np.array(ast.literal_eval(lines[gwline]))
    gb = np.array(ast.literal_eval(lines[gbline]))
    cw = np.array(ast.literal_eval(lines[cwline]))
    cb = np.array(ast.literal_eval(lines[cbline]))

    gw = gw.transpose(1,0)
    cw = cw.transpose(1,0)

    gwout = open('gw' + str(i + 1) + '.txt', 'w')
    gbout = open('gb' + str(i + 1) + '.txt', 'w')
    cwout = open('cw' + str(i + 1) + '.txt', 'w')
    cbout = open('cb' + str(i + 1) + '.txt', 'w')

    gwout.write(str(gw.tolist()))
    gbout.write(str(gb.tolist()))
    cwout.write(str(cw.tolist()))
    cbout.write(str(cb.tolist()))

    gwout.flush()
    gbout.flush()
    cwout.flush()
    cbout.flush()

    gwout.close()
    gbout.close()
    cwout.close()
    cbout.close()

for i in range(1):
    wline = 13 + 2 * i
    bline = 14 + 2 * i

    w = np.array(ast.literal_eval(lines[wline]))
    b = np.array(ast.literal_eval(lines[bline]))

    w = w.transpose(1,0)

    wout = open('w' + str(i + 1 + 3) + '.txt', 'w')
    bout = open('b' + str(i + 1 + 3) + '.txt', 'w')

    wout.write(str(w.tolist()))
    bout.write(str(b.tolist()))

    wout.flush()
    bout.flush()

    wout.close()
    bout.close()

input.close()
