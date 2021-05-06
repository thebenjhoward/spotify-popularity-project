

attribute_names = ["artists_count", "artist_popularity", "release_year", "duration", "explicit",
        "danceability", "energy", "tempo", "loudness", "time_signature", "popularity"]

def generate_columns(song_data, labels):
    """ Generates a dictionary of all values of each attribute

    Args:
        song_data(dict<str,list>): spotify songs dataset loaded.

    Returns:
        cols(dict): A dictionary of each column indexed by the attribute name. I would
            recommend deleting the song data dictionary if possible here
    """
    cols = {}
    for attrib in attribute_names[:-1]:
        cols[attrib] = []

    cols[attribute_names[-1]] = labels

    for song in song_data:
        for i, attrib in enumerate(attribute_names[:-1]):
            cols[attrib].append(song[i])


    return cols

def get_frequency(arr):
    """Gets the frequency of each value in a given array and returns as
    a dictionary indexed by each value

    Args:
        arr(list of obj): List of values to get the frequency of

    Returns:
        freq(dict): Dictionary containing the total occurances of each 
        value in the list
    """
    freq = {}
    for i in arr:
        if(i not in freq):
            freq[i] = 0
        freq[i] += 1
    
    return freq

def get_top_freq(freq_table, n):
    sorted_vals = sorted(freq_table.items(), key=lambda x: x[1])
    return sorted_vals[:n]