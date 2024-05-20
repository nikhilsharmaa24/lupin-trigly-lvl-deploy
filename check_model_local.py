import pickle

check_model = pickle.load(open('svr_model.pkl', 'rb'))
print(check_model.predict([[100, 120, 140]]))