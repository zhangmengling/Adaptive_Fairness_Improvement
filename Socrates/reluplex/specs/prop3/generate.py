def get_template():
    template = """{
  "model": {
    "shape": "(1,5)",
    "bounds": "[(-0.3035,-0.2986), (-0.0095,0.0095), (0.4934,0.5), (0.3,0.5), (0.3,0.5)]",
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
  "assert": "(FA x . TRUE => arg_min(x) != 0)"
}
    """

    return template


def get_nnet():
    nnet = ['reluplex/nnet/nnet_1_1', 'reluplex/nnet/nnet_1_2', 'reluplex/nnet/nnet_1_3', \
        'reluplex/nnet/nnet_1_4', 'reluplex/nnet/nnet_1_5', 'reluplex/nnet/nnet_1_6', \
        'reluplex/nnet/nnet_2_1', 'reluplex/nnet/nnet_2_2', 'reluplex/nnet/nnet_2_3', \
        'reluplex/nnet/nnet_2_4', 'reluplex/nnet/nnet_2_5', 'reluplex/nnet/nnet_2_6', \
        'reluplex/nnet/nnet_2_7', 'reluplex/nnet/nnet_2_8', 'reluplex/nnet/nnet_2_9', \
        'reluplex/nnet/nnet_3_1', 'reluplex/nnet/nnet_3_2', 'reluplex/nnet/nnet_3_3', \
        'reluplex/nnet/nnet_3_4', 'reluplex/nnet/nnet_3_5', 'reluplex/nnet/nnet_3_6', \
        'reluplex/nnet/nnet_3_7', 'reluplex/nnet/nnet_3_8', 'reluplex/nnet/nnet_3_9', \
        'reluplex/nnet/nnet_4_1', 'reluplex/nnet/nnet_4_2', 'reluplex/nnet/nnet_4_3', \
        'reluplex/nnet/nnet_4_4', 'reluplex/nnet/nnet_4_5', 'reluplex/nnet/nnet_4_6', \
        'reluplex/nnet/nnet_4_7', 'reluplex/nnet/nnet_4_8', 'reluplex/nnet/nnet_4_9', \
        'reluplex/nnet/nnet_5_1', 'reluplex/nnet/nnet_5_2', 'reluplex/nnet/nnet_5_3', \
        'reluplex/nnet/nnet_5_4', 'reluplex/nnet/nnet_5_5', 'reluplex/nnet/nnet_5_6', \
        'reluplex/nnet/nnet_5_7', 'reluplex/nnet/nnet_5_8', 'reluplex/nnet/nnet_5_9']

    return nnet


def get_prop():
    prop = ['prop3_nnet_1_1.json', 'prop3_nnet_1_2.json', 'prop3_nnet_1_3.json', \
        'prop3_nnet_1_4.json', 'prop3_nnet_1_5.json', 'prop3_nnet_1_6.json', \
        'prop3_nnet_2_1.json', 'prop3_nnet_2_2.json', 'prop3_nnet_2_3.json', \
        'prop3_nnet_2_4.json', 'prop3_nnet_2_5.json', 'prop3_nnet_2_6.json', \
        'prop3_nnet_2_7.json', 'prop3_nnet_2_8.json', 'prop3_nnet_2_9.json', \
        'prop3_nnet_3_1.json', 'prop3_nnet_3_2.json', 'prop3_nnet_3_3.json', \
        'prop3_nnet_3_4.json', 'prop3_nnet_3_5.json', 'prop3_nnet_3_6.json', \
        'prop3_nnet_3_7.json', 'prop3_nnet_3_8.json', 'prop3_nnet_3_9.json', \
        'prop3_nnet_4_1.json', 'prop3_nnet_4_2.json', 'prop3_nnet_4_3.json', \
        'prop3_nnet_4_4.json', 'prop3_nnet_4_5.json', 'prop3_nnet_4_6.json', \
        'prop3_nnet_4_7.json', 'prop3_nnet_4_8.json', 'prop3_nnet_4_9.json', \
        'prop3_nnet_5_1.json', 'prop3_nnet_5_2.json', 'prop3_nnet_5_3.json', \
        'prop3_nnet_5_4.json', 'prop3_nnet_5_5.json', 'prop3_nnet_5_6.json', \
        'prop3_nnet_5_7.json', 'prop3_nnet_5_8.json', 'prop3_nnet_5_9.json']

    return prop


def main():
    template = get_template()
    nnet = get_nnet()
    prop = get_prop()

    for i in range(42):
        nn = nnet[i]
        pp = prop[i]

        out = open(pp, 'w')

        tt = template % {"path": nn}

        out.write(tt)
        out.flush()
        out.close()


if __name__ == '__main__':
    main()
