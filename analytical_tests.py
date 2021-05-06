#!env python3
import multiprocessing
import threading
import pickle
import mysklearn.testing as testing
import mysklearn.myutils as myutils
from altsklearn.myclassifiers import MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.myrandomforestclassifier import MyRandomForestClassifier

attribs = [ 
    "artist_count", 
    "artist_popularity", 
    "release_year", 
    "duration", 
    "is_explicit", 
    "danceability",
    "energy",
    "tempo",
    "loudness", 
    "time_sig"
]
class_name = "popularity"

class_bins = [
    (0,11),
    (11,float('inf'))
]
class_labels = ["0-10", "11-100"]

bins = {
    "artist_count": [
        0,
        [(1,1), (2,3), (3,float('inf'))],
        ["1", "2-3", "4+"]
    ],
    "artist_popularity": [
        1,
        [(0,20), (20,40),(40,60),(60,80),(80,float('inf'))],
        ["0-20", "20-40", "40-60", "60-80", "80-100"]
    ],
    "release_year": [
        2,
        # [(float("-inf"), 1930), (1930,1950), (1950,1960), (1960,1970), (1970,1980), (1980,1990), (1990,2000), (2000,2010), (2010,2015), (2015,float('inf'))],
        # ["Before 1930", "30s-40s", "50s", "60s", "70s", "80s", "90s", "00s", "Early 10s", "Late 10s"]
        [(float("-inf"), 1950), (1950,1970), (1970,1990), (1990,2000), (2000,2010), (2010,2015), (2015,float('inf'))],
        ["Before 1950", "50s-60s", "70s-80s", "90s", "00s", "Early 10s", "Late 10s"]
    ],
    "duration": [
        3,
        [(0, 120), (120, 180), (180,240), (240, 300), (300, float('inf'))],
        ["<2:00", "2:00-3:00", "3:00-4:00", "4:00-5:00", ">5:00"]
    ],
    "dancibility": [
        5,
        # [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,float('inf'))],
        # ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        [(0,0.6),(0.6,float('inf'))],
        ["0-0.6", "0.6-1"]
    ],
    "energy": [
        6,
        # [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,float('inf'))],
        # ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        [(0,0.8),(0.8,float('inf'))],
        ["0-0.8", "0.8-1"]
    ],
    "tempo": [
        7,
        # [(float('-inf'), 60), (60,90), (90,110), (110, 130), (130,150), (150,180), (180,float('inf'))],
        # ["<60bpm", "60-90bpm,", "90-110bpm", "110-130bpm", "130-150bpm", "150-180bpm", ">180bpm"]
        [(float('-inf'), 60), (60,90), (90,150), (150,float('inf'))],
        ["<60bpm", "60-90bpm,", "90-150bpm", ">150bpm"]
    ],
    "loudness": [
        8,
        [(float('-inf'), -15), (-15,-10), (-10,-5), (-5,-3),(-3,float('inf'))],
        ["very quiet", "quiet", "soft", "loud", "very loud"]
    ]
}


def discretize_attribs(X, y):
    for _att, (i, abins, labs) in bins.items():
        #print(att)
        testing.bin_data(X, abins, labs, index=i)
    
    testing.bin_data(y, class_bins, class_labels)


def load_data(num=3):
    """ Loads the pickled data to begin testing for attribute selection
    
    Returns:
        instances(list of list): All attributes of each instance excluding the class label
        labels(list of obj): the popularity (not discretized) for each instance
    """

    with open('data/slices/slice0%i.pickle' % num, 'rb') as fp:
        return pickle.load(fp)

def run_analysis(X, y, labels, classifier_type, subsets, prog_queue, data_queue, proc_queue, pid):

    # for knn, test 3, 5 and 7 neighbors
    # impractical due to http timeout most likely
    if(classifier_type == "knn"):
        prog_queue.put({'type': "update_total", 'val': len(subsets), 'src': 'knn-%i' % pid}) # add number of tasks to progress total

        for subset in subsets:
            acc, matrix = testing.validate_classifier(X, y, labels, subset, MyKNeighborsClassifier(n_neighbors=7))
            prog_queue.put({'type': 'inc', 'src': 'knn-%i' % pid}) # increment progress
            data_queue.put(testing.gen_validation_result([attribs[i] for i in subset], "n_neighbors=17", labels, acc, matrix)) # put result string in queue

    elif(classifier_type == "nb"):
        prog_queue.put({ 'type': "update_total", 'val': len(subsets), 'src': 'nb-%i' % pid })
        for subset in subsets:
            acc, matrix = testing.validate_classifier(X, y, labels, subset, MyNaiveBayesClassifier())
            prog_queue.put({'type': 'inc', 'src': 'nb-%i' % pid}) # increment progress
            data_queue.put(testing.gen_validation_result([attribs[i] for i in subset], "", labels, acc, matrix))

    elif(classifier_type == "dt"):
        prog_queue.put({ 'type': "update_total", 'val': len(subsets), 'src': 'dt-%i' % pid })
        for subset in subsets:
            acc, matrix = testing.validate_classifier(X, y, labels, subset, MyDecisionTreeClassifier())
            prog_queue.put({'type': 'inc', 'src': 'nb-%i' % pid}) # increment progress
            data_queue.put(testing.gen_validation_result([attribs[i] for i in subset], "", labels, acc, matrix))
    elif(classifier_type == "forest"):
        prog_queue.put({ 'type': "update_total", 'val': len(subsets), 'src': 'forest-%i' % pid })
        for subset in subsets:
            acc,matrix = testing.validate_classifier(X, y, labels, subset, MyRandomForestClassifier(8, 4, 4))
            prog_queue.put({'type': 'inc', 'src': 'forest-%i' % pid})
            data_queue.put(testing.gen_validation_result([attribs[i] for i in subset], "", labels, acc, matrix))

    
    proc_queue.put(pid)

def run_progress(prog_queue):
    complete, total = 0, 0
    while(True):
        msg = prog_queue.get()
        if(msg['type'] == 'update_total'):
            total += msg['val']
            print("\n>", msg['src'], "added", msg['val'], 'tasks', flush=True)
            print("\033[K\033[1000D", myutils.progress_frac(complete, total), " ", 
                    myutils.progress_bar(complete, total, width=50), sep='', end='', flush=True)
        elif(msg['type'] == 'inc'):
            complete += 1
            print("\033[K\033[1000D", myutils.progress_frac(complete, total), " ", 
                    myutils.progress_bar(complete, total, width=50), sep='', end='', flush=True)
        elif(msg['type'] == 'stop'):
            break


def good_classifier():
    attributes = [0, 1, 2, 5, 6, 8]
    X_data, y_data = load_data(num=3)
    discretize_attribs(X_data, y_data)

    Xs = []
    for inst in X_data:
        Xs.append([inst[i] for i in attributes])
    
    model = MyDecisionTreeClassifier()
    model.fit(Xs, y_data)

    with open('./model.pickle', 'wb') as fp:
        pickle.dump(model, fp)
    
    model.visualize_tree('final-tree.dot', 'tree.pdf', attribute_names=[attribs[i] for i in attributes], fmt='pdf')

    


def main():
    print("loading data...", end='')
    X_data, y_data = load_data(num=3)
    print("done")

    print("binning and normalizing...", end='')
    discretize_attribs(X_data, y_data)
    print(X_data[0])
    print('done')

    next_pid = 0
    nb_chunks = list(testing.chunk_subsets(testing.elem_subsets(range(len(attribs)), range(3,11)), 16))
    dt_chunks = list(testing.chunk_subsets(testing.elem_subsets(range(len(attribs)), range(3,11)), 16))
    # forest_chunks = list(testing.chunk_subsets(testing.elem_subsets(range(len(attribs)), range(9,11)), 12))
    
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()    # queue for managing process updates from multiple processes
    nb_queue = manager.Queue()          # queue for storing return from naive bayes
    dt_queue = manager.Queue()
    # forest_queue = manager.Queue()
    process_queue = manager.Queue()     # queue for keeping track of finished processes to join

    tasks = []

    # create processes
    for chunk in nb_chunks:
        tasks.append(multiprocessing.Process(target=run_analysis,
                args=(X_data, y_data, class_labels, "nb", chunk, progress_queue, nb_queue, process_queue, next_pid)))
        next_pid += 1

    for chunk in dt_chunks:
        tasks.append(multiprocessing.Process(target=run_analysis,
                args=(X_data, y_data, class_labels, "dt", chunk, progress_queue, dt_queue, process_queue, next_pid)))
        next_pid += 1

    # for chunk in forest_chunks:
    #     tasks.append(multiprocessing.Process(target=run_analysis,
    #             args=(X_data, y_data, class_labels, "forest", chunk, progress_queue, forest_queue, process_queue, next_pid)))
    #     next_pid += 1
    
    prog_thread = threading.Thread(target=run_progress, args=(progress_queue,))


    prog_thread.start()

    # start and join processes       
    for i in range(min(16,len(tasks))):
        tasks[i].start()

    curr_index = min(16,len(tasks))
    returned = 0

    while(returned < len(tasks)):
        ret = process_queue.get()
        
        tasks[ret].join()

        returned += 1
        if(curr_index < len(tasks)):
            tasks[curr_index].start()
            curr_index += 1


    progress_queue.put({"type": "stop"})
    prog_thread.join()

    fp = open('./output.log', 'w')
    print("\nNAIVE BAYES:")
    while(not nb_queue.empty()):
        item = nb_queue.get()
        print(item)
        print(item, file=fp)
    print("\nDECISION TREES")
    while(not dt_queue.empty()):
        item = dt_queue.get()
        print(item)
        print(item, file=fp)
    # print("\nRANDOM FORESTS")
    # while(not forest_queue.empty()):
    #     item = forest_queue.get()
    #     print(item)
    #     print(item, file=fp)


if __name__ == "__main__":
    main()




