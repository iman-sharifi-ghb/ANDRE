#!/usr/bin/env python3
"""
Third-Party Data Preprocessing Script
This script processes relational data files and generates propositionalized training datasets
for logical rule learning. It performs classification of relations, generates valid predicates,
and creates trainable feature matrices.
"""

import os
import sys
import argparse
import logging
import pickle
import itertools
import time
import numpy as np
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def classifier(raw_dataset_path, variable_number, original_data_path, t_relation=''):
    """
    Extract and classify relations from raw dataset file.
    
    Parses a relational dataset file and extracts:
    - All relations and their arities
    - Objects and their classifications
    - Predicates with their probabilities
    
    Args:
        raw_dataset_path: Path to the raw dataset file (.nl format)
        variable_number: Number of variables in the learning task
        original_data_path: Directory to save processed data
        t_relation: Target relation name
        
    Returns:
        tuple: (variable_objects dict, relation names list, relation dict, 
                arity_relation dict, target_arity)
    """
    all_rlation = {}
    save_all_relation = {}
    all_predicate = []
    all_object = {}
    relation = {}
    arity_relation = {}
    pro = {}
    
    # Parse the dataset file
    with open(raw_dataset_path, 'r') as f:
        new_single_line = f.readline()
        while new_single_line:
            # Skip test data
            if '#' in new_single_line and 'TEST' in new_single_line:
                new_single_line = f.readline()
                continue
            
            one_perd = []
            
            # Extract predicate name and objects
            predicate = new_single_line[0:new_single_line.index(')')]
            
            # Extract probability if present
            if '#' in new_single_line:
                if 'T.#' in new_single_line or 'T#' in new_single_line:
                    probability = 1
                else:
                    probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')]
                    probability = probability.replace('.', '').replace(' ', '').replace('#', '')
                    probability = float('0.' + probability)
            else:
                probability = 1
            
            # Parse relation and objects
            single_line = predicate.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            one_perd.append(probability)
            all_predicate.append(one_perd)
            
            # Initialize object and relation tracking
            if first_obj not in all_object:
                all_object[first_obj] = set([])
            if second_obj not in all_object:
                all_object[second_obj] = set([])
            if relation_name not in all_rlation:
                all_rlation[relation_name] = [set([]), set([])]
            relation[relation_name] = []
            arity_relation[relation_name] = 1  # Initial arity for all relations is 1
            
            new_single_line = f.readline()
            pro[predicate + ')'] = probability
    
    all_relation_list = list(all_rlation)
    
    # Save all relation names
    for i in all_relation_list:
        save_all_relation[i] = i
    
    with open(os.path.join(original_data_path, 'all_relation_dic.dt'), 'wb') as f:
        pickle.dump(save_all_relation, f)
    
    with open(os.path.join(original_data_path, 'all_relation_dic.txt'), 'w') as f:
        print(str(save_all_relation), file=f)
    
    # Classify objects by their relation positions
    for pred in all_predicate:
        first_string = str(all_relation_list.index(pred[0])) + '-1'
        second_string = str(all_relation_list.index(pred[0])) + '-2'
        all_object[pred[1]].add(first_string)
        all_object[pred[2]].add(second_string)
    
    # Build relation tuples
    for pred in all_predicate:
        one_tuple = (pred[1], pred[2])
        relation[pred[0]].append(one_tuple)
        
        # Check arity of predicates
        if pred[1] != pred[2] and arity_relation[pred[0]] == 1:
            arity_relation[pred[0]] = 2
    
    # Create variable-object classification
    variable_objects = {}
    for i in range(variable_number):
        variable_objects[i] = set([])
    
    target_arity = arity_relation[t_relation]
    
    if target_arity == 2:
        # For binary predicates, classify first two variables based on target relation
        for pred in all_predicate:
            if pred[0] == t_relation:
                variable_objects[0].add(pred[1])
                variable_objects[1].add(pred[2])
        # Remaining variables can be any object
        for obj in all_object:
            for variable_name in range(2, variable_number):
                variable_objects[variable_name].add(obj)
    elif target_arity == 1:
        # For unary predicates, all variables have the same classification
        for obj in all_object:
            for variable_name in range(variable_number):
                variable_objects[variable_name].add(obj)
    
    # Save relation entities
    with open(os.path.join(original_data_path, 'relation_entities.dt'), 'wb') as f:
        pickle.dump(relation, f)
    
    # Save probabilities
    with open(os.path.join(original_data_path, 'pro.dt'), 'wb') as f:
        pickle.dump(pro, f)
    
    with open(os.path.join(original_data_path, 'pro.txt'), 'w') as f:
        print(pro, file=f)
    
    return variable_objects, list(relation.keys()), relation, arity_relation, target_arity


def show_process(step, all_process, current_process, start_time, note=''):
    """
    Display progress information with estimated time remaining.
    
    Args:
        step: Number of steps between progress updates
        all_process: Total number of steps
        current_process: Current step number
        start_time: Start time of the process
        note: Additional note to display
    
    Returns:
        float: Current time
    """
    process = float(current_process * 100 / all_process)
    step_time = float((time.time() - start_time) / 60.0)
    rest_running_time = float(((all_process - current_process) / step) * step_time)
    
    if rest_running_time > 60:
        print(f"{note} Percent: {process:.2f}% Current step: {int(current_process)} "
              f"Total steps: {int(all_process)} Remain running time: {rest_running_time/60:.2f} (hours).", 
              end='\r')
    else:
        print(f"{note} Percent: {process:.2f}% Current step: {int(current_process)} "
              f"Total steps: {int(all_process)} Remain running time: {rest_running_time:.2f} (minutes).", 
              end='\r')
    
    return time.time()


def get_all_valide_predicate(all_substitution, relation, all_variable_permination, t_relation, 
                             t_index, number_all_sub=0):
    """
    Identify valid predicates by checking which predicates have non-zero values.
    
    This implements the propositionalization method mentioned in the paper:
    removes predicates that are always zero across all substitutions.
    
    Args:
        all_substitution: All possible variable substitutions
        relation: Dictionary of relations and their tuples
        all_variable_permination: Variable permutations for each relation
        t_relation: Target relation name
        t_index: Index of target predicate
        number_all_sub: Total number of substitutions
        
    Returns:
        tuple: (valid_index list, template dict) or (-1, -1) if too large
    """
    logger.info(f'Total number of all substitutions: {number_all_sub}')
    
    # Memory constraint: reject if too large
    if number_all_sub >= 1.5e7:
        return -1, -1
    
    flag_predicates = {}
    relation_index = 0
    relation_name = list(relation.keys())
    target_predicate_index = relation_name.index(t_relation)
    
    logger.info(f"Target predicate index: {target_predicate_index}")
    
    # Initialize flags for all predicates
    for i in relation:
        flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
        relation_index += 1
    
    sub_index = 0
    start_time = time.time()
    
    # Check all substitutions to find which predicates are ever true
    for i in all_substitution:
        current_relation_index = 0
        
        for relation_name in relation:
            current_data = relation[relation_name]
            current_predicate_index = 0
            current_permination = all_variable_permination[relation_name]
            
            for j in current_permination:
                first_variable = j[0]
                second_variable = j[1]
                target_tuple = (i[first_variable], i[second_variable])
                
                if target_tuple in current_data:
                    flag_predicates[current_relation_index][current_predicate_index] = 1
                
                current_predicate_index += 1
            current_relation_index += 1
        
        sub_index += 1
        if sub_index % 2000 == 0:
            start_time = show_process(2000, number_all_sub, sub_index, start_time, 
                                     note='[Find Valid Features]')
    
    logger.info(f"All predicate info: {flag_predicates}")
    
    # Build list of valid predicate indices
    valid_index = []
    number_of_predicate = 0
    for i in flag_predicates:
        info = flag_predicates[i]
        for j in info:
            if j == 1 and number_of_predicate != t_index:
                valid_index.append(number_of_predicate)
            number_of_predicate += 1
    
    logger.info(f"Valid predicate indices: {valid_index}")
    
    # Create template for valid predicates
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1
    
    logger.info(f"Valid template: {template}")
    logger.info("Check Valid Predicate Success")
    
    return valid_index, template


def make_all_predicate(relation, relation_variable_permutations):
    """
    Generate list of all first-order features in the dataset.
    
    Args:
        relation: Dictionary of relations
        relation_variable_permutations: Variable permutations for each relation
        
    Returns:
        list: All predicate variable pairs
    """
    a_p_a_v = []
    for relation_name in relation:
        current_permination = relation_variable_permutations[relation_name]
        for variable_pair in current_permination:
            a_p_a_v.append(variable_pair)
    return a_p_a_v


def make_small_data(all_substitution, relation, relation_variable_permutations, template, 
                   pro, t_relation, train_label, original_data_path, actual_n_substitutation):
    """
    Generate trainable data for small datasets.
    
    Generates all trainable data in the format: 
    X: [[predicate_values]] for each relation
    y: [target_predicate_value]
    
    Args:
        all_substitution: All variable substitutions
        relation: Dictionary of relations
        relation_variable_permutations: Variable permutations for each relation
        template: Valid predicate template
        pro: Probability values for facts
        t_relation: Target relation name
        train_label: Dictionary to store training labels
        original_data_path: Path to save data
        actual_n_substitutation: Number of substitutions
        
    Returns:
        tuple: (valid_substitution count, total_substitution count)
    """
    times_substitution = 0
    valid_substitution = 0
    start_time = time.time()
    
    for single_substitution in all_substitution:
        relation_index = 0
        wrong_flag = True  # Flag for constraint: if all inputs are 0, label should not be 1
        all_body_boolean_value = {}
        
        # Evaluate each relation for this substitution
        for relation_name in relation:
            y_one_data = []
            data = relation[relation_name]
            current_permination = relation_variable_permutations[relation_name]
            
            for variable_pair in current_permination:
                string_tuple = tuple([single_substitution[m] for m in variable_pair])
                
                if string_tuple in data:
                    # Build symbolic relational fact
                    current_fact = f"{relation_name}({string_tuple[0]},{string_tuple[1]})"
                    prob_value = pro[current_fact]
                    y_one_data.append(prob_value)
                else:
                    y_one_data.append(0)
            
            all_body_boolean_value[relation_index] = y_one_data
            relation_index += 1
        
        times_substitution += 1
        
        # Check if any body predicate is true
        check_ind = 0
        for i in all_body_boolean_value:
            label_list = []
            for acq in template[check_ind]:
                label_list.append(all_body_boolean_value[i][acq])
            if 1 in label_list:
                wrong_flag = False
                break
            check_ind += 1
        
        # Evaluate target predicate
        target_tuple = (single_substitution[0], single_substitution[1])
        
        # Skip if target is true but all body predicates are false
        if target_tuple in relation[t_relation] and wrong_flag == True:
            continue
        
        # Add all non-target predicates to training dataset
        for i in all_body_boolean_value:
            train_label[i].append(all_body_boolean_value[i])
        
        # Add target predicate feature
        if target_tuple in relation[t_relation]:
            current_fact = f"{t_relation}({target_tuple[0]},{target_tuple[1]})"
            current_fact_pro = pro[current_fact]
            train_label[relation_index].append([current_fact_pro])
        else:
            train_label[relation_index].append([0])
        
        valid_substitution += 1
        
        if times_substitution % 5000 == 0:
            start_time = show_process(5000, actual_n_substitutation, times_substitution, 
                                     start_time, note='[Make Trainable data]')
    
    train_label[len(relation) + 1] = train_label[len(relation)]
    
    logger.info(f"Generated data shape: {len(train_label)}, {len(train_label[0])}, "
                f"{len(train_label[1])}, {len(train_label[2])}")
    
    # Prepare X and y
    final_x = []
    for i in range(len(train_label) - 1):
        final_x.append(train_label[i])
    
    new_train_label = train_label[len(train_label) - 1]
    
    # Save to files
    x_file = os.path.join(original_data_path, 'x_train.dt')
    y_file = os.path.join(original_data_path, 'y_train.dt')
    
    with open(x_file, 'wb') as xf:
        pickle.dump(final_x, xf)
    
    with open(y_file, 'wb') as yf:
        pickle.dump(new_train_label, yf)
    
    return valid_substitution, times_substitution


def main_processing(dataset_file, t_relation='', path_name='', original_data_path='', 
                   variable_depth=1, input_dataset_path=''):
    """
    Main processing pipeline for data generation.
    
    Args:
        dataset_file: Name of the dataset file
        t_relation: Target relation name
        path_name: Path to save intermediate files
        original_data_path: Path to save original data
        variable_depth: Depth of variable relationships
        input_dataset_path: Path to read input dataset from
        
    Returns:
        tuple: (ALL_objects list, target_arity, target_predicate) or (-1, -1, -1) if failed
    """
    variable_number = variable_depth + 2
    
    # Construct full input path
    input_file_path = os.path.join(input_dataset_path, dataset_file)
    
    # Step 1: Classify relations and objects
    variable_objects, relation_name, relation, arity_relation, target_arity = classifier(
        input_file_path, 
        variable_number, 
        original_data_path, 
        t_relation
    )
    
    logger.info(f'Begin generating data with {t_relation} as head predicate.')
    
    # Build variable classification
    variable_class = {}
    for i in variable_objects:
        variable_class[i] = list(variable_objects[i])
    
    # Collect all objects
    ALL_objects = []
    for i in variable_class:
        l = variable_class[i]
        for j in l:
            if j not in ALL_objects:
                ALL_objects.append(j)
    
    # Create entity dictionary
    entities_dic_name_number = {}
    for i in ALL_objects:
        entities_dic_name_number[i] = ALL_objects.index(i)
    
    logger.info(f"All relations in dataset: {relation}")
    
    # Generate all substitutions
    variable_value = variable_class.values()
    all_substitution_number = 1
    for i in variable_value:
        all_substitution_number *= len(i)
    
    all_substitution = list(itertools.product(*variable_value))
    actual_n_substitutation = all_substitution_number
    
    # Generate variable permutations for each relation
    relation_variable_permutations = {}
    for i in arity_relation:
        if arity_relation[i] == 1:
            # Unary relations: same variable in both positions
            empty_list = [(variable_index, variable_index) for variable_index in range(variable_number)]
            relation_variable_permutations[i] = empty_list
        else:
            # Binary relations: all permutations of variables
            relation_variable_permutations[i] = list(itertools.permutations(range(variable_number), 2))
    
    # Find target predicate index
    find_index = 0
    for i in range(len(relation_name)):
        for j in relation_variable_permutations[relation_name[i]]:
            if arity_relation[t_relation] == 1 and j == (0, 0) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = f"{t_relation}(X,X)"
            elif arity_relation[t_relation] == 2 and j == (0, 1) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = f"{t_relation}(X,Y)"
            find_index += 1
    
    logger.info(f'Target predicate index: {t_index}')
    logger.info(f'All variable permutations: {relation_variable_permutations}')
    
    # Save variable permutations
    with open(os.path.join(path_name, 'relation_variable.dt'), 'wb') as f:
        pickle.dump(relation_variable_permutations, f)
    
    with open(os.path.join(path_name, 'relation_variable.txt'), 'w') as f:
        print(relation_variable_permutations, file=f)
    
    # Initialize training labels
    train_label = {}
    for i in range(len(relation)):
        train_label[i] = []
    
    target_predicate_index = len(relation)
    train_label[target_predicate_index] = []
    
    logger.info(f"Pre-operation on label dataset: {train_label}")
    
    # Step 2: Find or generate valid predicates
    try:
        with open(os.path.join(path_name, 'valid_index.dt'), "rb") as f:
            res = pickle.load(f)
            valid_index = res['valid_index']
            template = res['template']
            logger.info(f"Loaded valid predicates: {res}")
    except FileNotFoundError:
        valid_index, template = get_all_valide_predicate(
            all_substitution=all_substitution,
            relation=relation,
            all_variable_permination=relation_variable_permutations,
            t_relation=t_relation,
            t_index=t_index,
            number_all_sub=actual_n_substitutation
        )
        
        if valid_index == -1 and template == -1:
            return -1, -1, -1
        
        # Save valid predicates
        res = {'valid_index': valid_index, 'template': template}
        with open(os.path.join(path_name, 'valid_index.dt'), 'wb') as f:
            pickle.dump(res, f)
        
        with open(os.path.join(path_name, 'valid_index.txt'), 'w') as f:
            print(str(res), file=f)
        
        logger.info("Saved template successfully")
    
    # Load probabilities
    with open(os.path.join(path_name, 'pro.dt'), "rb") as f:
        pro = pickle.load(f)
    
    # Step 3: Generate trainable data
    valid_substitution, times_substitution = make_small_data(
        all_substitution, relation, relation_variable_permutations, template, 
        pro, t_relation, train_label, original_data_path, actual_n_substitutation
    )
    
    # Save meta-information
    with open(os.path.join(original_data_path, 'number_valid_substitution.txt'), 'w') as vf:
        vf.write(str(valid_substitution))
        logger.info(f'{valid_substitution} (valid) / {times_substitution} (all)')
    
    # Step 4: Save all variable permutations
    logger.info("Saving all variable permutations...")
    a_p_a_v = make_all_predicate(relation, relation_variable_permutations)
    
    with open(os.path.join(original_data_path, 'predicate.txt'), 'w') as vf:
        vf.write(str(a_p_a_v))
    
    with open(os.path.join(original_data_path, 'predicate.list'), 'wb') as vf:
        pickle.dump(a_p_a_v, vf)
    
    logger.info("Save success")
    
    # Save all objects
    all_obj_file = os.path.join(original_data_path, 'all_ent.dt')
    with open(all_obj_file, 'wb') as f:
        pickle.dump(ALL_objects, f)
    
    return ALL_objects, target_arity, target_predicate


def data_assemble(dataset_name, dataset_file, predicate, variable_depth):
    """
    Assemble data and generate meta-information.
    
    Args:
        dataset_name: Name of dataset (e.g., 'nations', 'umls')
        dataset_file: Name of dataset file (e.g., 'nations.nl')
        predicate: Target predicate name
        variable_depth: Depth of variable relationships
        
    Returns:
        int: 0 if successful
        
    Raises:
        NameError: If data generation takes too long
    """
    # Construct paths
    dataset_dir = os.path.join('datasets', dataset_name)
    input_path = os.path.join(dataset_dir, dataset_file)
    output_dir = os.path.join(dataset_dir, predicate)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Reading dataset from: {input_path}")
    logger.info(f"Saving results to: {output_dir}")
    
    _, target_arity, head_pre = main_processing(
        dataset_file, 
        t_relation=predicate, 
        path_name=output_dir,
        original_data_path=output_dir, 
        variable_depth=variable_depth,
        input_dataset_path=dataset_dir
    )
    
    if target_arity == -1 and head_pre == -1:
        logger.warning("Data generation checking valid predicate time is too long!")
        raise NameError('RUNTOOLONG')
    
    # Save meta-information
    meta_info = {'target_arity': target_arity, "head_pre": head_pre}
    
    with open(os.path.join(output_dir, "meta_info.dt"), 'wb') as f:
        pickle.dump(meta_info, f)
    
    with open(os.path.join(output_dir, "meta_info.txt"), 'w') as f:
        print(meta_info, file=f)
    
    logger.info(f"Generated all trainable data successfully with variable depth: {variable_depth}")
    
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Third-Party Data Preprocessing for Relational Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset (e.g., nations, umls)')
    parser.add_argument('--dataset_file', type=str, required=True,
                        help='Name of the dataset file (e.g., nations.nl)')
    parser.add_argument('--predicate', type=str, required=True,
                        help='Target predicate/relation name')
    parser.add_argument('--variable_depth', type=int, default=1,
                        help='Depth of variable relationships (usually 2 for binary predicates)')
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Third-Party Data Preprocessing")
    logger.info("="*80)
    logger.info(f"Dataset Name: {args.dataset_name}")
    logger.info(f"Dataset File: {args.dataset_file}")
    logger.info(f"Target Predicate: {args.predicate}")
    logger.info(f"Variable Depth: {args.variable_depth}")
    logger.info(f"Input Directory: datasets/{args.dataset_name}/")
    logger.info(f"Output Directory: datasets/{args.dataset_name}/{args.predicate}/")
    logger.info("="*80)
    
    # Run data assembly
    try:
        result = data_assemble(
            dataset_name=args.dataset_name,
            dataset_file=args.dataset_file,
            predicate=args.predicate,
            variable_depth=args.variable_depth
        )
        
        if result == 0:
            logger.info("="*80)
            logger.info("Data preprocessing completed successfully!")
            logger.info(f"Output files saved to: datasets/{args.dataset_name}/{args.predicate}/")
            logger.info("="*80)
        
    except NameError as e:
        logger.error(f"Error during processing: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
