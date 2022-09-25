import matplotlib.pyplot as plt
import numpy as np
import os

class Plot():
    def __init__(self, dataset_name, sens_attr, processing_name):
        self.dataset_name = dataset_name
        self.sens_attr = sens_attr
        self.processing_name = processing_name

    def plot_acc_metric(self, orig_metrics, improved_metrics, metric_name):
        orig_best_ind = np.argmax(orig_metrics['bal_acc'])
        improved_best_ind = np.argmax(improved_metrics['bal_acc'])
        # orig_acc = orig_metrics['bal_acc'][orig_best_ind]
        orig_acc = orig_metrics['acc'][orig_best_ind]
        orig_metric = abs(orig_metrics[metric_name][orig_best_ind])
        # improved_acc = improved_metrics['bal_acc'][improved_best_ind]
        improved_acc = improved_metrics['accplot_abs_acc_multi_metric'][improved_best_ind]
        improved_metric = abs(improved_metrics[metric_name][improved_best_ind])
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        title = str(self.processing_name) + " " + self.dataset_name + "-" + self.sens_attr
        plt.title(title)
        plt.xlabel("accuracy")
        plt.ylabel(metric_name)
        plt.plot(orig_acc, orig_metric, 'bo', label="original model")
        plt.plot(improved_acc, improved_metric, 'ro', label="improved model")
        print("-->original", orig_acc, orig_metric)
        print("-->improved", improved_acc, improved_metric)
        plt.annotate("", xy=(improved_acc, improved_metric), xytext=(orig_acc, orig_metric),
                     arrowprops=dict(arrowstyle="->"))
        plt.show()

    # multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
    def plot_acc_multi_metric(self, orig_metrics, improved_metrics, multi_metric_names):
        plt.clf()
        orig_best_ind = np.argmax(orig_metrics['acc'])
        improved_best_ind = np.argmax(improved_metrics['acc'])
        orig_acc = orig_metrics['acc'][orig_best_ind]
        plt.xlim(0.5, 1.0)
        plt.ylim(0, 0.5)
        title = str(self.processing_name) + "-" + self.dataset_name + "-" + self.sens_attr
        plt.title(title)
        plt.xlabel("accuracy")
        # plt.xlabel("balanced accuracy")
        plt.ylabel("metrics")
        colors = ['black', 'darkorange', 'green']
        for i in range(0, len(multi_metric_names)):
            metric_name = multi_metric_names[i]
            orig_metric = abs(orig_metrics[metric_name][orig_best_ind])
            improved_acc = improved_metrics['acc'][improved_best_ind]
            if min(orig_acc, improved_acc) < 0.5:
                plt.xlim(min(orig_acc, improved_acc) - 0.1, 1.0)
            improved_metric = abs(improved_metrics[metric_name][improved_best_ind])
            if max(orig_metric, improved_metric) > 0.5:
                plt.ylim(0, max(orig_metric, improved_metric)+0.3)
            if i == 0:
                plt.plot(orig_acc, orig_metric, 'bo', label="original model")
                plt.plot(improved_acc, improved_metric, 'ro', label = "improved model")
            else:
                plt.plot(orig_acc, orig_metric, 'bo')
                plt.plot(improved_acc, improved_metric, 'ro')
            # plt.annotate(text="", xy=(improved_acc, improved_metric), xytext=(orig_acc, orig_metric),
            #              arrowprops=dict(arrowstyle="->"))
            plt.plot([improved_acc, orig_acc], [improved_metric, orig_metric], color=colors[i], label=metric_name)
            # plt.text((improved_acc+orig_acc)/2, (improved_metric+orig_metric)/2, metric_name)
        plt.legend()
        plt.show()
        file_path = os.path.abspath(os.path.dirname(__file__))
        file_name = file_path + "/plotting_result/" + title + ".png"
        plt.savefig(file_name)

    #### plot all metrics for single or multiple sensitive attributes for one improvement method and one dataset
    def plot_abs_acc_all_metric(self, orig_uni_metrics1, improved_uni_metrics1, orig_uni_metrics2, improved_uni_metrics2,
        orig_multi_metrics, improved_multi_metrics, **kwargs):

        try:
            uni_metric_names = kwargs['metric_names']
            # uni_metric_names = ['stat_par_diff', 'group']
        except:
            uni_metric_names = ['stat_par_diff', 'group', 'causal']

        sens_attrs = self.sens_attr
        # orig_best_ind = np.argmax(orig_multi_metrics['acc'])
        orig_best_ind = 0
        colors = ['darkorange', 'green', 'black']
        markers = ['^', 'v', 's']

        uni_abs_acc1 = improved_uni_metrics1['acc'][orig_best_ind] - orig_uni_metrics1['acc'][orig_best_ind]
        uni_abs_acc2 = improved_uni_metrics2['acc'][orig_best_ind] - orig_uni_metrics2['acc'][orig_best_ind]
        multi_abs_acc = improved_multi_metrics['acc'][orig_best_ind] - orig_multi_metrics['acc'][orig_best_ind]
        max_acc = max(abs(uni_abs_acc1), abs(uni_abs_acc2), abs(multi_abs_acc))
        if max_acc > 0.5:
            plt.xlim(-max_acc - 0.01, max_acc + 0.01)
        else:
            plt.xlim(-max_acc - 0.01, max_acc + 0.01)

        max_metrics = []
        for i in range(0, len(uni_metric_names)):
            plt.axhline(y=0.0, color='red', linestyle=(0, (5, 10)))
            plt.axvline(x=0.0, color='red', linestyle=(0, (5, 10)))
            plt.xlabel("accuracy")
            # plt.xlabel("balanced accuracy")
            plt.ylabel("metrics")

            metric_name = uni_metric_names[i]
            print("-->metric_name", metric_name)
            uni_orig_metric1 = abs(orig_uni_metrics1[metric_name][orig_best_ind])
            uni_improved_metric1 = abs(improved_uni_metrics1[metric_name][orig_best_ind])
            abs_metric1 = uni_improved_metric1 - uni_orig_metric1
            uni_orig_metric2 = abs(orig_uni_metrics2[metric_name][orig_best_ind])
            uni_improved_metric2 = abs(improved_uni_metrics2[metric_name][orig_best_ind])
            abs_metric2 = uni_improved_metric2 - uni_orig_metric2
            multi_orig_metric = abs(orig_multi_metrics[metric_name][orig_best_ind])
            multi_improved_metric = abs(improved_multi_metrics[metric_name][orig_best_ind])
            abs_multi_metric = multi_improved_metric - multi_orig_metric

            max_metric = max(abs(abs_metric1), abs(abs_metric2), abs(abs_multi_metric))
            max_metrics.append(max_metric)

            if i == 0:
                plt.scatter(x=uni_abs_acc1, y=abs_metric1, c=colors[0], marker=markers[i], label=sens_attrs[0])
                plt.scatter(x=uni_abs_acc2, y=abs_metric2, c=colors[1], marker=markers[i], label=sens_attrs[1])
                plt.scatter(x=multi_abs_acc, y=abs_multi_metric, c=colors[2], marker=markers[i],
                            label=str(sens_attrs[0])+","+str(sens_attrs[1]))
            else:
                plt.scatter(x=uni_abs_acc1, y=abs_metric1, c=colors[0], marker=markers[i])
                plt.scatter(x=uni_abs_acc2, y=abs_metric2, c=colors[1], marker=markers[i])
                plt.scatter(x=multi_abs_acc, y=abs_multi_metric, c=colors[2], marker=markers[i])

        low_metric = max(0, min(max_metrics))
        up_metric = min(0, max(max_metrics))
        plt.ylim(-max(max_metrics) - 0.01, max(max_metrics) + 0.01)
        title = str(self.processing_name) + "-" + self.dataset_name + "-" + str(self.sens_attr)
        plt.title(title)
        plt.legend()
        # plt.show()
        file_path = os.path.abspath(os.path.dirname(__file__))
        file_name = file_path + "/plotting_result/" + title + ".png"
        plt.savefig(file_name)

    def plot_abs_acc_all_methods(self, all_orig_uni_metrics, all_improved_uni_metrics, all_orig_multi_metrics, all_improved_multi_metrics):
        # uni_metric_names = ['group', 'causal']
        uni_metric_names = ['group']
        method_colors = ['deepskyblue', 'steelblue', 'blue', 'salmon', 'orange',
                         'orangered', 'firebrick', 'lime', 'limegreen', 'green']
        orig_best_ind = 0
        markers = ['^', 'v', 's']

        position = [(1, 2, 1), (1, 2, 2, ), (1, 2, 3)]

        for j in range(0, len(self.dataset_name)):
            dataset = self.dataset_name[j]
            one_orig_uni_metrics = all_orig_uni_metrics[j]
            print("-->one_orig_uni_metrics", one_orig_uni_metrics)
            one_improved_uni_metrics = all_improved_uni_metrics[j]
            # one_orig_multi_metrics = all_orig_multi_metrics[j]
            # one_improved_multi_metrics = all_improved_multi_metrics[j]
            sens_attrs = self.sens_attr[j]
            for p in range(0, len(self.processing_name)):
                process_name = self.processing_name[p]
                orig_uni_metrics = one_orig_uni_metrics[p]
                print("-->orig_uni_metrics", orig_uni_metrics)
                improved_uni_metrics = one_improved_uni_metrics[p]
                # orig_multi_metrics = one_orig_multi_metrics[p]
                # improved_multi_metrics = one_improved_multi_metrics[p]

                for i in range(0, len(uni_metric_names)):
                    # plt = plt.figure()
                    plt.clf()
                    plt.axhline(y=0.0, color='red', linestyle=(0, (5, 10)))
                    plt.axvline(x=0.0, color='red', linestyle=(0, (5, 10)))
                    plt.xlabel("accuracy")
                    # plt.xlabel("balanced accuracy")
                    plt.ylabel("metrics")

                    plt.scatter(x=1, y=1, c='black', marker=markers[0], label=sens_attrs[0])
                    plt.scatter(x=1, y=1, c='black', marker=markers[1], label=sens_attrs[1])
                    plt.scatter(x=1, y=1, c='black', marker=markers[2],
                                label=str(sens_attrs[0]) + "," + str(sens_attrs[1]))

                    metric_name = uni_metric_names[i]
                    print("-->metric_name", metric_name)

                    max_metrics = []
                    max_accs = []
                    processing_name = self.processing_name[j]
                    print("-->processing_name", processing_name)
                    print("--orig", orig_uni_metrics)
                    print("-->improved", improved_uni_metrics)
                    orig_uni_metrics1 = orig_uni_metrics[0]
                    improved_uni_metrics1 = improved_uni_metrics[0]
                    orig_uni_metrics2 = orig_uni_metrics[1]
                    improved_uni_metrics2 = improved_uni_metrics[1]
                    # orig_multi_metrics0 = orig_multi_metrics[j]
                    # improved_multi_metrics0 = improved_multi_metrics[j]

                    uni_abs_acc1 = improved_uni_metrics1['acc'][orig_best_ind] - orig_uni_metrics1['acc'][
                        orig_best_ind]
                    uni_abs_acc2 = improved_uni_metrics2['acc'][orig_best_ind] - orig_uni_metrics2['acc'][
                        orig_best_ind]
                    # multi_abs_acc = improved_multi_metrics0['acc'][orig_best_ind] - orig_multi_metrics0['acc'][orig_best_ind]
                    max_accs.append(uni_abs_acc1)
                    max_accs.append(uni_abs_acc2)
                    # max_accs.append(multi_abs_acc)

                    try:
                        uni_orig_metric1 = abs(orig_uni_metrics1[metric_name][orig_best_ind])
                        uni_improved_metric1 = abs(improved_uni_metrics1[metric_name][orig_best_ind])
                        print(uni_improved_metric1, uni_orig_metric1)
                        abs_metric1 = uni_improved_metric1 - uni_orig_metric1
                        uni_orig_metric2 = abs(orig_uni_metrics2[metric_name][orig_best_ind])
                        uni_improved_metric2 = abs(improved_uni_metrics2[metric_name][orig_best_ind])
                        abs_metric2 = uni_improved_metric2 - uni_orig_metric2
                        # multi_orig_metric = abs(orig_multi_metrics0[metric_name][orig_best_ind])
                        # multi_improved_metric = abs(improved_multi_metrics0[metric_name][orig_best_ind])
                        # abs_multi_metric = multi_improved_metric - multi_orig_metric

                        max_metrics.append(abs_metric1)
                        max_metrics.append(abs_metric2)
                        # max_metrics.append(abs_multi_metric)

                        plt.scatter(x=uni_abs_acc1, y=abs_metric1, c=method_colors[j], marker=markers[0],
                                    alpha=3 / 5,
                                    label=process_name)
                        plt.scatter(x=uni_abs_acc2, y=abs_metric2, c=method_colors[j], marker=markers[1],
                                    alpha=3 / 5)
                        # plt.scatter(x=multi_abs_acc, y=abs_multi_metric, c=method_colors[j], marker=markers[2],alpha=3 / 5)
                    except:
                        print("No value for metric:", metric_name)
                        print("-->processing name:", processing_name)

                plt.xlim(-max(abs(min(max_accs)), abs(max(max_accs))) - 0.02,
                         max(abs(min(max_accs)), abs(max(max_accs))) + 0.02)
                plt.ylim(-max(abs(min(max_metrics)), abs(max(max_metrics))) - 0.02,
                         max(abs(min(max_metrics)), abs(max(max_metrics))) + 0.02)
                # title = metric_name + " metric testing on single,multiple sensitive attributes (" + self.dataset_name + ")"
                title = dataset + " Dataset"
                plt.title(title)

                if j == 0:
                    plt.subplot(1, 3, 1)
                elif j == 1:
                    plt.subplot(1, 3, 2)
                else:
                    plt.subplot(1, 3, 3)

                    # plt.legend()
                    # plt.show()
                    # file_path = os.path.abspath(os.path.dirname(__file__))
                    # file_name = file_path + "/plotting_result/" + title + ".png"
                    # plt.savefig(file_name)
                    
        plt.suptitle("Group Metric on Single Sensitive Feature")
        plt.show()
        file_path = os.path.abspath(os.path.dirname(__file__))
        file_name = file_path + "/plotting_result/group_metric_uni.png"
        plt.savefig(file_name)




    ##### compare metrics for single/multiple discrimination metrics
    ##### also include o: for group metric; ^:for causal metric
    # def plot_abs_acc_multi_metric(self, orig_uni_metrics1, improved_uni_metrics1, orig_uni_metrics2, improved_uni_metrics2,
    #     orig_multi_metrics, improved_multi_metrics):
    def plot_abs_acc_multi_metric(self, orig_uni_metrics, improved_uni_metrics, orig_multi_metrics, improved_multi_metrics):
        # orig_uni_metrics, improved_uni_metrics, orig_multi_metrics, improved_multi_metrics

        # print("-->orig_uni_metrics1", orig_uni_metrics)
        # print("-->orig_multi_metrics", orig_multi_metrics)
        # print("-->improved_multi_metrics", improved_multi_metrics)

        sens_attrs = self.sens_attr
        uni_metric_names = ['group', 'causal']
        # uni_metric_names = ['group']
        # orig_best_ind = np.argmax(orig_multi_metrics['acc'])
        orig_best_ind = 0
        # colors = ['darkorange', 'green', 'black']
        method_colors = ['deepskyblue', 'blue', 'salmon', 'orange',
                         'orangered', 'firebrick', 'lime', 'limegreen', 'green']
        markers = ['^', 'v', 's']
        # markers = ["v", '^', 's']

        # if isinstance(self.processing_name, list):
        #     for method_index in range(0, len(self.processing_name)):
        #         processing_name = self.processing_name[method_index]

        if isinstance(orig_uni_metrics[0], dict):
            print("-->only one improvement method")
            orig_uni_metrics1 = [orig_uni_metrics[0]]
            improved_uni_metrics1 = [improved_uni_metrics[0]]
            orig_uni_metrics2 = [orig_uni_metrics[1]]
            improved_uni_metrics2 = [improved_uni_metrics[1]]
            orig_multi_metrics = [orig_multi_metrics]
            improved_multi_metrics = [improved_multi_metrics]

        for i in range(0, len(uni_metric_names)):
            # plt = plt.figure()
            plt.clf()
            plt.axhline(y=0.0, color='red', linestyle=(0, (5, 10)))
            plt.axvline(x=0.0, color='red', linestyle=(0, (5, 10)))
            plt.xlabel("accuracy")
            # plt.xlabel("balanced accuracy")
            plt.ylabel("metrics")

            plt.scatter(x=1, y=1, c='black', marker=markers[0], label=sens_attrs[0])
            plt.scatter(x=1, y=1, c='black', marker=markers[1], label=sens_attrs[1])
            plt.scatter(x=1, y=1, c='black', marker=markers[2], label=str(sens_attrs[0]) + "," + str(sens_attrs[1]))

            plt.legend(loc="upper right")

            metric_name = uni_metric_names[i]
            print("-->metric_name", metric_name)

            max_metrics = []
            max_accs = []
            for j in range(0, len(orig_uni_metrics)):
                processing_name = self.processing_name[j]
                print("-->processing_name", processing_name)
                print("-->improved_uni_metrics")
                print("--orig", orig_uni_metrics[j])
                print("-->improved", improved_uni_metrics[j])
                orig_uni_metrics1 = orig_uni_metrics[j][0]
                improved_uni_metrics1 = improved_uni_metrics[j][0]
                orig_uni_metrics2 = orig_uni_metrics[j][1]
                improved_uni_metrics2 = improved_uni_metrics[j][1]
                orig_multi_metrics0 = orig_multi_metrics[j]  # for race,sex attributes
                improved_multi_metrics0 = improved_multi_metrics[j]

                uni_abs_acc1 = improved_uni_metrics1['acc'][orig_best_ind] - orig_uni_metrics1['acc'][orig_best_ind]
                uni_abs_acc2 = improved_uni_metrics2['acc'][orig_best_ind] - orig_uni_metrics2['acc'][orig_best_ind]
                multi_abs_acc = improved_multi_metrics0['acc'][orig_best_ind] - orig_multi_metrics0['acc'][orig_best_ind]
                max_accs.append(uni_abs_acc1)
                max_accs.append(uni_abs_acc2)
                max_accs.append(multi_abs_acc)

                try:
                    uni_orig_metric1 = abs(orig_uni_metrics1[metric_name][orig_best_ind])
                    uni_improved_metric1 = abs(improved_uni_metrics1[metric_name][orig_best_ind])
                    print(uni_improved_metric1, uni_orig_metric1)
                    abs_metric1 = uni_improved_metric1 - uni_orig_metric1
                    uni_orig_metric2 = abs(orig_uni_metrics2[metric_name][orig_best_ind])
                    uni_improved_metric2 = abs(improved_uni_metrics2[metric_name][orig_best_ind])
                    abs_metric2 = uni_improved_metric2 - uni_orig_metric2
                    multi_orig_metric = abs(orig_multi_metrics0[metric_name][orig_best_ind])
                    multi_improved_metric = abs(improved_multi_metrics0[metric_name][orig_best_ind])
                    abs_multi_metric = multi_improved_metric - multi_orig_metric

                    max_metrics.append(abs_metric1)
                    max_metrics.append(abs_metric2)
                    max_metrics.append(abs_multi_metric)

                    plt.scatter(x=uni_abs_acc1, y=abs_metric1, c=method_colors[j], marker=markers[0], alpha=4/5, label=processing_name)
                    plt.scatter(x=uni_abs_acc2, y=abs_metric2, c=method_colors[j], marker=markers[1], alpha=4/5)
                    plt.scatter(x=multi_abs_acc, y=abs_multi_metric, c=method_colors[j], marker=markers[2], alpha=4/5)

                except:
                    print("No value for metric:", metric_name)
                    print("-->processing name:", processing_name)


            print("-->max_metrics", max(max_metrics), min(max_metrics))
            print(max_accs)
            print("-->max accs", max(max_accs), min(max_accs))
            plt.xlim(-max(abs(min(max_accs)), abs(max(max_accs))) - 0.02, max(abs(min(max_accs)), abs(max(max_accs))) + 0.02)
            plt.ylim(-max(abs(min(max_metrics)), abs(max(max_metrics))) - 0.02, max(abs(min(max_metrics)), abs(max(max_metrics))) + 0.02)
            # title = metric_name + " metric testing on single,multiple sensitive attributes (" + self.dataset_name + ")"
            title = self.dataset_name
            plt.title(title)

            # plt.legend(ncol=4, bbox_to_anchor=(0.8, -0.2), borderaxespad=0, numpoints=1, fontsize=10)
            # plt.subplots_adjust(bottom=0.3)

            plt.legend()
            # plt.show()
            file_path = os.path.abspath(os.path.dirname(__file__))
            file_name = file_path + "/plotting_result/" + metric_name + title + ".png"
            print("-->file_name", file_name)
            # file_name = file_path + "/plotting_result/" + title + ".png"
            plt.savefig(file_name)


    def show_oneset_attribute(self, orig_uni_metrics, improved_uni_metrics, metric_names, best_ind, sens_attrs, colors):
        plt.clf()
        plt.axhline(y=0.0, color='red', linestyle=(0, (5, 10)))
        plt.axvline(x=0.0, color='red', linestyle=(0, (5, 10)))
        plt.xlabel("accuracy")
        # plt.xlabel("balanced accuracy")
        plt.ylabel("metrics")
        markers = ['o', 'x']

        if isinstance(orig_uni_metrics[0], dict):
            print("-->isinstance(orig_uni_metrics[0], dict)")
            orig_uni_metrics = [metric for metric in orig_uni_metrics]
            improved_uni_metrics = [metric for metric in improved_uni_metrics]

        plt.scatter(x=1, y=1, c='black', marker=markers[0], label="group metric")
        plt.scatter(x=1, y=1, c='black', marker=markers[1], label="causal metric")

        # plt.legend(loc="lower right")

        max_metrics = []
        max_accs = []
        for j in range(0, len(orig_uni_metrics)):
            processing_name = self.processing_name[j]
            orig_metrics = orig_uni_metrics[j]
            improved_metrics = improved_uni_metrics[j]
            abs_acc = improved_metrics['acc'][best_ind] - orig_metrics['acc'][best_ind]
            max_accs.append(abs(abs_acc))

            for i in range(0, len(metric_names)):
                #### metric_names=['group', 'causal']
                metric_name = metric_names[i]
                try:
                    orig_metric = abs(orig_metrics[metric_name][best_ind])
                    improved_metric = abs(improved_metrics[metric_name][best_ind])
                    abs_metric = improved_metric - orig_metric

                    max_metrics.append(abs(abs_metric))

                    if i == 0:
                        plt.scatter(x=abs_acc, y=abs_metric, c=colors[j], marker=markers[i], alpha=3/5,
                                    label=processing_name)
                    else:
                        plt.scatter(x=abs_acc, y=abs_metric, c=colors[j], marker=markers[i], alpha=3/5)
                except:
                    continue

        print("-->max metrics", max(max_metrics), min(max_metrics))
        print("-->max accs", max(max_accs), min(max_accs))
        # compas: 0.545, credit: 0.267, adult: 0.404
        plt.xlim(-max(max_accs) - 0.02, max(max_accs) + 0.02)
        plt.ylim(-max(max_metrics) - 0.02, max(max_metrics) + 0.02)
        plt.xlabel("accuracy")
        plt.ylabel("metrics")
        # title = str(self.processing_name) + "-" + self.dataset_name + "-" + sens_attrs
        # title = "metric testing on group,causal metrics-" + str(sens_attrs) + "(" + self.dataset_name + ")"
        if isinstance(sens_attrs, list):
            if sens_attrs == ["race", "sex"]:
                sens_attrs = str(sens_attrs[1] + ", " + sens_attrs[0])
            else:
                sens_attrs = str(sens_attrs[0] + ", " + sens_attrs[1])
        title = self.dataset_name + "-" + sens_attrs
        print("-->title", title)
        plt.title(title)
        plt.legend()
        plt.show()
        file_path = os.path.abspath(os.path.dirname(__file__))
        file_name = file_path + "/plotting_result/" + title + ".png"
        plt.savefig(file_name)

    ##### compare metrics for group/individual(causal) discrimination metrics
    def plot_abs_acc_individual_metric(self, orig_uni_metrics, improved_uni_metrics, orig_multi_metrics, improved_multi_metrics):
        print("-->plot_abs_acc_individual_metric")
        sens_attrs = self.sens_attr
        uni_metric_names = ['group', 'causal']
        multi_metric_names = ['multi_group', 'multi_causal']
        # orig_best_ind = np.argmax(orig_multi_metrics['acc'])
        orig_best_ind = 0
        # colors = ['darkorange', 'green']
        method_colors = ['deepskyblue', 'steelblue', 'blue', 'salmon', 'orange',
                         'orangered', 'firebrick', 'lime', 'limegreen', 'green']
        markers = ['^', 'v', ',']

        orig_uni_metrics1 = [metric[0] for metric in orig_uni_metrics]
        improved_uni_metrics1 = [metric[0] for metric in improved_uni_metrics]
        orig_uni_metrics2 = [metric[1] for metric in orig_uni_metrics]
        improved_uni_metrics2 = [metric[1] for metric in improved_uni_metrics]

        print("-->orig_uni_metrics1", orig_uni_metrics1)
        print("-->improved_uni_metrics1", improved_uni_metrics1)
        print("-->orig_uni_metrics2", orig_uni_metrics2)
        print("-->improved_uni_metrics2", improved_uni_metrics2)

        # for metric1: race/sex as sensitive attribute (only consider orig_uni_metrics1 and improved_uni_metrics1)
        self.show_oneset_attribute(orig_uni_metrics1, improved_uni_metrics1, uni_metric_names, orig_best_ind,
                                   sens_attrs[0], method_colors)

        # for metric2: sex/race as sensitive attribute
        self.show_oneset_attribute(orig_uni_metrics2, improved_uni_metrics2, uni_metric_names, orig_best_ind,
                                   sens_attrs[1], method_colors)

        # for multi_metric: race and sex as sensitive attributes
        self.show_oneset_attribute(orig_multi_metrics, improved_multi_metrics, uni_metric_names, orig_best_ind,
                                   sens_attrs, method_colors)

    def plot_one_abs_acc_individual_metric(self, orig_uni_metrics, improved_uni_metrics):
        print("-->plot_abs_acc_individual_metric")
        sens_attrs = self.sens_attr
        uni_metric_names = ['group', 'causal']
        multi_metric_names = ['multi_group', 'multi_causal']
        # orig_best_ind = np.argmax(orig_multi_metrics['acc'])
        orig_best_ind = 0
        # colors = ['darkorange', 'green']
        method_colors = ['deepskyblue', 'steelblue', 'blue', 'salmon', 'orange',
                         'orangered', 'firebrick', 'lime', 'limegreen', 'green']
        markers = ['^', 'v', ',']

        orig_uni_metrics1 = [metric[0] for metric in orig_uni_metrics]
        improved_uni_metrics1 = [metric[0] for metric in improved_uni_metrics]

        print("-->orig_uni_metrics1", orig_uni_metrics1)
        print("-->improved_uni_metrics1", improved_uni_metrics1)

        # for metric1: race/sex as sensitive attribute (only consider orig_uni_metrics1 and improved_uni_metrics1)
        self.show_oneset_attribute(orig_uni_metrics1, improved_uni_metrics1, uni_metric_names, orig_best_ind,
                                   sens_attrs[0], method_colors)

    def plot_multi_thre(self, x, x_name, y_left, y_left_name, y_right, y_right_name):
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.plot(x, y_left)
        ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
        ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.set_ylim(0.5, 0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, y_right, color='r')
        ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
        if 'DI' in y_right_name:
            ax2.set_ylim(0., 0.7)
        else:
            ax2.set_ylim(-0.25, 0.1)

        best_ind = np.argmax(y_left)
        ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        # plt.show()

    def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.plot(x, y_left)
        ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
        ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.set_ylim(0.5, 0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, y_right, color='r')
        ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
        if 'DI' in y_right_name:
            ax2.set_ylim(0., 0.7)
        else:
            ax2.set_ylim(-0.25, 0.1)

        best_ind = np.argmax(y_left)
        ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        # plt.show()
