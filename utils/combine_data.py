import pickle

def main():
    data = {
        'speech_data': [],
        'vision_data': [],
        'object_names': [],
        'instance_names': [],
    }

    with open('speech_features.pkl', 'rb') as fin:
        speech = pickle.load(fin)

    with open('gld_vision_text_tensors.pkl', 'rb') as fin:
        gld = pickle.load(fin)

    for i, o, v in zip(gld['instance_names'], gld['object_names'], gld['vision_data']):
        # remove frame number    
        instance_name = '_'.join(i.split('_')[:-1])
        #print(i, instance_name, o)

        indices = [i for i, v in enumerate(speech['instance_name']) if v == instance_name]
        #print(instance_name, indices)

        for j in indices:
            data['speech_data'].append(speech['mfcc'][j])
            data['vision_data'].append(v)
            data['object_names'].append(o)
            data['instance_names'].append(instance_name)

    print(len(data['object_names']))
    
    with open('gld_speech_vision_tensors.pkl', 'wb') as fout:
        pickle.dump(data, fout)

    print(f'Wrote {len(data["object_names"])} data points to file gld_speech_vision_tensors.pkl')

if __name__ == '__main__':
    main()
