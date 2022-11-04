import os

def main():
    for k1 in range(1,6):
        for k2 in range(1,10):
            file = open('ACASXU_run2a_' + str(k1) + '_' + str(k2) + '_batch_2000.nnet', 'r')
            path = './nnet_' + str(k1) + '_' + str(k2) + '/'
            os.mkdir(path)

            pathw = path + 'weights/'
            pathb = path + 'bias/'

            os.mkdir(pathw)
            os.mkdir(pathb)

            lines = file.readlines()

            for i in range(6):
                wsrt = 10 + 100 * i
                wend = 60 + 100 * i

                bsrt = 60 + 100 * i
                bend = 110 + 100 * i

                wfile = open(pathw + 'w' + str(i + 1) + '.txt', 'w')
                bfile = open(pathb + 'b' + str(i + 1) + '.txt', 'w')

                w = ''
                b = ''

                for j in range(wsrt, wend):
                    line = lines[j]
                    if j == wsrt:
                        w += '[['
                    else:
                        w += '['
                    w += line[:-2]
                    if j == wend - 1:
                        w += ']]'
                    else:
                        w += '], '

                wfile.write(w)
                wfile.flush()
                wfile.close()

                for j in range(bsrt, bend):
                    line = lines[j]
                    if j == bsrt:
                        b += '['
                    b += line[:-2]
                    if j == bend - 1:
                        b += ']'
                    else:
                        b += ','

                bfile.write(b)
                bfile.flush()
                bfile.close()

            wfile = open(pathw + 'w7.txt', 'w')
            bfile = open(pathb + 'b7.txt', 'w')

            w = ''
            b = ''

            wsrt = 610
            wend = 615

            bsrt = 615
            bend = 620

            for j in range(wsrt, wend):
                line = lines[j]
                if j == wsrt:
                    w += '[['
                else:
                    w += '['
                w += line[:-2]
                if j == wend - 1:
                    w += ']]'
                else:
                    w += '], '

            wfile.write(w)
            wfile.flush()
            wfile.close()

            for j in range(bsrt, bend):
                line = lines[j]
                if j == bsrt:
                    b += '['
                b += line[:-2]
                if j == bend - 1:
                    b += ']'
                else:
                    b += ','

            bfile.write(b)
            bfile.flush()
            bfile.close()


if __name__ == '__main__':
    main()
