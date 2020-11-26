MPST: A Corpus of Movie Plot Synopses with Tags
========================================================================
Sudipta Kar, Suraj Maharjan, A. Pastor L\'opez-Monroy and Thamar Solorio
------------------------------------------------------------------------

This corpus provides written plot synopses of 14,828 movies and 71 tags 
multi-label associations with them.

Directory Listing
------------------


MPST
│   README.txt
│   LICENCE
|	id_title_tags.tsv : Tab seperated file contains IMDb ID, title, and 
|	tags for movies
|	train_ids.txt : IMDb ID for movies in the training set
|	test_ids.txt : IMDb id for movies in the test set
│   
└───final_plots_wiki_imdb_combined
│   │   imdb_id_list.txt : Lists the 14,828 IMDb IDs of the movies 
│   │   synopsis_sources.tsv : Lists the source (IMDb/Wikipedia) of the
|	|	synopses 
│   │
│   └───raw : Synopses are stored in separate files for each IMDb ID as 
|		they were collected. 
│   └───cleaned : All but the alphaneumeric characters are removed from 
|		the synopses. Each line of the files contains the words of one 
|		sentence.
│       
│   
└───tag_assignment_data
    │   tag_list.txt : Lists the 71 unique tags in the dataset
    │   tag_group_label_to_number_of_movies.tsv : Lists the tags and 
    |	number of movies for them
    │   movie_to_number_of_label.json : A dictionary where Key: IMDb ID, 
    |	Value: Number of tags assigned to the ID
    │   movie_to_label_name.json : A dictionary where Key: IMDb ID, 
    |	Value: A list of tags for the ID
    │   label_names_to_movie.json : A dictionary where Key: tag, Value: 
    |	A list of movies tagged with the key tag.


Contact
========
Sudipta Kar
skar3@uh.edu
