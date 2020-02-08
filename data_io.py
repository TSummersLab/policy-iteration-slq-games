import pickle
import os

def filenamer(problem_data_id):
    problem_data_id_str = '%010d' % problem_data_id
    filename = 'problem_data_'+problem_data_id_str+'.pkl'
    return filename

def pather(problem_data_id):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    problem_data_directory = 'problem_data'
    filename = filenamer(problem_data_id)
    path = os.path.join(location, problem_data_directory, filename)
    return path

def save_problem_data(problem_data_id, problem_data):
    path_out = pather(problem_data_id)
    file_out = open(path_out, 'wb')
    pickle.dump(problem_data, file_out)
    file_out.close()
    return

def load_problem_data(problem_data_id):
    path_in = pather(problem_data_id)
    file_in = open(path_in, 'rb')
    problem_data = pickle.load(file_in)
    file_in.close()
    return problem_data