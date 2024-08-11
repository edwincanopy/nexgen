from aijson import Flow
import asyncio
import os
import re
import ast

async def add_columns(raw_data, new_columns, info_about_user):
    '''

    Takes in raw data, adds columns using an LLM and outputs updated data.
    
    INPUT:
      - raw_data: a list of dictionaries with 2 keys each. 
                  Example: [{'Name':'Ben Launders', 'Comments':'Goes to Stanford, has 3 startups in med-tech, millionaire.}, {'Name':'Kim Flowers', 'Comments':'Billionaire in NY, quite formal, runs a biotech corporation which owns 7 companies}]
      
      - new_columns: a list of names of new columns to be added 
                  Example: ["Industries", "How do you know them", "Role(s)", "Commonalities"]   

      - info_about_user: string containing info about the user
                  Example: 'My name is Humpty. I am an egg'           
    
    OUTPUT: 
      - updated_data: a list of dictionaries with added columns
    
    '''

    updated_data = raw_data
    
    return updated_data


def relevant_data_finder(data, prompt):
    '''

    Takes in the entire dataset and the prompt and returns the most relevant rows. 

    INPUT: 
      - data:         a list of dictionaries 
                      Example: [{'Name':'Ben Launders', 'Comments':'Goes to Stanford, has 3 startups in med-tech, millionaire.}, {'Name':'Kim Flowers', 'Comments':'Billionaire in NY, quite formal, runs a biotech corporation which owns 7 companies}]
            
      - prompt:       prompt entered by user
                      Example: 'I want to break into the edtech space. Find me the most relevant people' 


    OUTPUT: 
      - output_data:  a list of dictionaries containing the most relevant rows. There must be another column in this data called 'Reason' which tells you why this person was picked

    ''' 
    
    # Modify code here 
    output_data = data 
    for row in output_data:
        row['Reason'] = 'Just coz'

    return output_data


def write_message(chosen_person, chosen_comments, chosen_reason, more_thoughts, prompt, info_about_user):

    filename = 'text_style_transfer.ai.yaml'
    request = prompt
    info_about_person_to_contact = f'Name: {chosen_person}. Comments:{chosen_comments}'

    message_format = 'LinkedIn'
    number_of_words = '50-70'

    message = asyncio.run(helper_function_for_write_message(filename,request, info_about_user, info_about_person_to_contact, message_format, number_of_words, more_thoughts))

    return message




def find_dictionary_by_name(data, name):
    # Normalize the input name by stripping whitespace and converting to lowercase
    normalized_name = name.strip().lower()

    # Iterate over each dictionary in the list
    for entry in data:
        # Normalize the name field in the dictionary
        dict_name = entry.get('Name', '').strip().lower()
        
        # Check if the normalized input name matches the normalized dictionary name
        if normalized_name == dict_name:
            return entry
    
    # Return None or a message if no match is found
    return None

async def helper_function_for_write_message(filename, request, info_about_user, info_about_person_to_contact, message_format, number_of_words, more_thoughts):

    flow = Flow.from_file(filename)
    flow = flow.set_vars(request=request, info_about_user=info_about_user, info_about_person_to_contact=info_about_person_to_contact, message_format=message_format, number_of_words=number_of_words, more_thoughts=more_thoughts)
    result = await flow.run()

    # alternatively, INSTEAD of running it, stream it
    async for result in flow.stream():
        return result