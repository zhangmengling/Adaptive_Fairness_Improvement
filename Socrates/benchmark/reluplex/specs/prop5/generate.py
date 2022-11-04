def get_template():
    template = """{
  "model": {
    "shape": "(1,5)",
    "bounds": "[(-0.3243,-0.3218), (0.0318,0.0637), (-0.5,-0.4992), (-0.5,-0.2273), (-0.5,-0.1667)]",
    "layers": [
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w1.txt",
        "bias": "benchmark/%(path)s/bias/b1.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w2.txt",
        "bias": "benchmark/%(path)s/bias/b2.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w3.txt",
        "bias": "benchmark/%(path)s/bias/b3.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w4.txt",
        "bias": "benchmark/%(path)s/bias/b4.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w5.txt",
        "bias": "benchmark/%(path)s/bias/b5.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w6.txt",
        "bias": "benchmark/%(path)s/bias/b6.txt",
        "func": "relu"
      },
      {
        "type": "linear",
        "weights": "benchmark/%(path)s/weights/w7.txt",
        "bias": "benchmark/%(path)s/bias/b7.txt",
        "func": "softmax"
      }
    ]
  },
  "assert": "(FA x . TRUE => arg_min(x) = 4)"
}
    """

    return template


def get_nnet():
    nnet = ['reluplex/nnet/nnet_1_1']

    return nnet


def get_prop():
    prop = ['prop5_nnet_1_1.json']

    return prop


def main():
    template = get_template()
    nnet = get_nnet()
    prop = get_prop()

    for i in range(1):
        nn = nnet[i]
        pp = prop[i]

        out = open(pp, 'w')

        tt = template % {"path": nn}

        out.write(tt)
        out.flush()
        out.close()


if __name__ == '__main__':
    main()
