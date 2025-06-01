import numpy as np
from pprint import pprint
import requests
from sklearn.cluster import KMeans
import logging
import itertools

def generate_uniform_points_with_endpoints(start, end, D):

    start = np.array(start)
    end = np.array(end)

    direction = end - start
    total_length = np.linalg.norm(direction)
    unit_direction = direction / total_length

    num_segments = max(1, int(np.round(total_length / D)))

    distances = np.linspace(0, total_length, num_segments + 1)

    points = start + np.outer(distances, unit_direction)
    return points

def ordered_points_along_line(points, P1, P2, margin):

    points = np.array(points)
    P1 = np.array(P1)
    P2 = np.array(P2)

    line_vec = P2 - P1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        raise ValueError("P1 and P2 cannot be the same")

    unit_vec = line_vec / line_len

    rel_vecs = points - P1

    scalar_projs = rel_vecs @ unit_vec

    projections = P1 + np.outer(scalar_projs, unit_vec)

    perp_dists = np.linalg.norm(points - projections, axis=1)

    mask = perp_dists <= margin
    filtered_points = points[mask]
    filtered_scalars = scalar_projs[mask]

    sorted_indices = np.argsort(filtered_scalars)
    ordered_points = filtered_points[sorted_indices]

    return ordered_points

def most_collinear_triplet(points):
    best_triplet = None
    min_cross = float('inf')
    best_unit_vector = None
    best_dist = None

    for triplet in itertools.combinations(points, 3):
        A, B, C = map(np.array, triplet)
        AB = B - A
        AC = C - A

        cross = np.abs(np.cross(AB, AC))
        if cross < min_cross:
            min_cross = cross
            best_triplet = triplet
            unit_vector = AB / np.linalg.norm(AB)
            best_unit_vector = unit_vector

    A, B, C = best_triplet[0], best_triplet[1], best_triplet[2]
    AB_norm = np.linalg.norm(B - A)
    AC_norm = np.linalg.norm(C - A)
    BC_norm = np.linalg.norm(C - B)
    #find the longest distance
    best_dist = max(AB_norm, AC_norm, BC_norm) /2

    best_triplet_hashable = map(tuple, best_triplet)  # Convert to tuples for immutability
    points = list(map(tuple, points))  # Convert to tuples for immutability
    leftover_points = set(points) - set(best_triplet_hashable)

    leftover_points = list(leftover_points)
    vector = np.array(leftover_points[0]) - np.array(leftover_points[1])
    left_over_unit_vector = vector / np.linalg.norm(vector)

    dot_product = np.dot(best_unit_vector, left_over_unit_vector)
    if abs(dot_product) < 0.15:  # Close to zero
        return best_triplet, min_cross, best_unit_vector, best_dist
    else:
        return None

def missing_trees_orchard(target_orchard_id, auth_header):
    url = "https://api.aerobotics.com/farming/surveys/"

    headers = {
        "accept": "application/json",
        "Authorization": auth_header # Change to environment variable with Docker secrets?
    }


    response = requests.get(url, headers=headers)

    # first check the resposne code for an auth error
    if response.status_code == 401:
        logging.error("Authorization successful")
        return {"error": "Authorization failed, please check your token."}

    response_json = response.json()

    results = response_json.get("results", [])

    orchard_id_to_survey_id = {}
    for survey in results:
        orchard_id = survey.get("orchard_id")
        survey_id = survey.get("id")
        if orchard_id and survey_id:
            orchard_id_to_survey_id[orchard_id] = survey_id

    survey_id = orchard_id_to_survey_id.get(target_orchard_id, None)
    if survey_id is None:
        logging.error(f"No survey found for orchard ID {target_orchard_id}")
        return {"error": f"No survey found for orchard ID {target_orchard_id}"}


    url = f"https://api.aerobotics.com/farming/surveys/{survey_id}/tree_surveys"

    headers = {
        "accept": "application/json",
        "Authorization": auth_header
    }

    response = requests.get(url, headers=headers)
    data = response.json()['results']

    collect_lngs_lats = [(tree['lng'], tree['lat']) for tree in data if 'lng' in tree and 'lat' in tree]


    trees_np = np.array(collect_lngs_lats)
    unit_vector_candidates = []
    unit_vector_candidates_losses = []
    collect_best_dists = []


    # first we need to find the axes of the grid-like orchard
    # there is a strong assumption that the trees are arranged in a grid-like pattern such that a tree's nearest neighbors are collinear
    for n in range(len(trees_np)):

        grid_n_diff = trees_np - trees_np[n] # these are now essentially vectors from the highlight point
        grid_n_euc_dists = np.linalg.norm(grid_n_diff, axis=1)
        nearest_indices = np.argsort(grid_n_euc_dists)[:5] # this should give us the tree and its 4 nearest neighbors
        nearest_trees = trees_np[nearest_indices]

        # any two points are collinear, but three points nominate a line, so let's find three collinear points
        ret = most_collinear_triplet(nearest_trees)
        if ret is None: # no collinear triplet found, skip
            continue
        most_collinear, min_cross, unit_vector_candidate, best_dist = ret
        collect_best_dists.append(best_dist)
        unit_vector_candidates.append(unit_vector_candidate)
        unit_vector_candidates_losses.append(min_cross) # the cross product is a measure of how collinear the points are, smaller is better

    unit_vector_candidates = np.array(unit_vector_candidates)
    unit_vector_candidates_losses = np.array(unit_vector_candidates_losses)
    collect_best_dists = np.array(collect_best_dists)
    
    # we do 4 clusters to get the two axes (both directions for each may be candidates)
    kmeans = KMeans(n_clusters=4, random_state=(hash('aerobotics') % len('aerobotics'))) 
    labels = kmeans.fit_predict(unit_vector_candidates)

    cluster0 = unit_vector_candidates[labels == 0]
    cluster1 = unit_vector_candidates[labels == 1]
    cluster2 = unit_vector_candidates[labels == 2]
    cluster3 = unit_vector_candidates[labels == 3]

    weights = 1 / (unit_vector_candidates_losses + 1e-8)  # TODO: test negative log?

    cluster0 = unit_vector_candidates[labels == 0]
    w0 = weights[labels == 0]
    dists0 = collect_best_dists[labels == 0]

    cluster1 = unit_vector_candidates[labels == 1]
    w1 = weights[labels == 1]
    dists1 = collect_best_dists[labels == 1]

    cluster2 = unit_vector_candidates[labels == 2]
    w2 = weights[labels == 2]
    dists2 = collect_best_dists[labels == 2]

    cluster3 = unit_vector_candidates[labels == 3]
    w3 = weights[labels == 3]
    dists3 = collect_best_dists[labels == 3]

    first_axis_est = np.average(cluster0, axis=0, weights=w0)
    second_axis_est = np.average(cluster1, axis=0, weights=w1)
    third_axis_est = np.average(cluster2, axis=0, weights=w2)
    fourth_axis_est = np.average(cluster3, axis=0, weights=w3)

    # find the first mostly-orthogonal pair of axes, any will do
    if np.abs(np.dot(first_axis_est, second_axis_est)) < 0.15:
        a1 = first_axis_est
        a1_dist = np.mean(dists0)
        a2 = second_axis_est
        a2_dist = np.mean(dists1)
    else:
        a1 = first_axis_est
        a1_dist = np.mean(dists0)
        a2 = third_axis_est
        a2_dist = np.mean(dists2)

    # a1 and a2 are now two "mostly" orthogonal axes of this grid-like orchard
    # could get cross product of any axis to get its orthogonal pair, but I think this allows for skewness of the grid

    # now we we find the missing trees along both of these axes
    missing_trees_memory = {}

    for axis, dist in zip([a1, a2], [a1_dist, a2_dist]):
        row_labels_mask = np.zeros(len(trees_np), dtype=np.int8)  # mask for trees that are along the row, all 0 (unassigned) at first

        r = 1
        while any(row_labels_mask == 0):  # while there are still unassigned trees
            unassigned_indices = np.where(row_labels_mask == 0)[0]  # get indices of unassigned trees
            n = unassigned_indices[0]  # take the first unassigned tree, but any will do

            line_start_x = trees_np[n][0]
            line_start_y = trees_np[n][1]
            line_start = np.array([line_start_x, line_start_y])
            line_end = line_start + axis # place axis on this tree

            # we need to have some idea of where a tree is expected given the axis and its average tree spacing
            ordered_points = ordered_points_along_line(trees_np, line_start, line_end, margin=dist/2)
            

            # assign all these points to the same row
            row_labels_mask[np.isin(trees_np, ordered_points).all(axis=1)] = r

            # now we generate tree expected locations along this line. Again, strong assumption that trees are evenly spaced along a row
            generated_uniform_points_along_line = generate_uniform_points_with_endpoints(ordered_points[0], ordered_points[-1], dist)
            r += 1  # increment row number
            
            # now we check if any of these points are missing from the trees
            for point in generated_uniform_points_along_line:
                dists_to_all_trees = np.linalg.norm(trees_np - point, axis=1)
                closest_tree = np.min(dists_to_all_trees)

                if closest_tree > dist/2:
                    if len(missing_trees_memory) == 0: # first missing tree
                        missing_trees_memory[0] = [point]
                    else: # maybe this missing location has been seen before?
                        # it has been seen before if is it within half of the dist
                        seen_trees = [tree_list[0] for tree_list in missing_trees_memory.values()]
                        dists_to_seen_trees = np.linalg.norm(seen_trees - point, axis=1)
                        if np.min(dists_to_seen_trees) < dist/2:
                            # this tree has been seen before, so we can add it to the existing list
                            argmin = np.argmin(dists_to_seen_trees)
                            missing_trees_memory[argmin].append(point)
                        else: # this is a new missing tree, make a new entry
                            missing_trees_memory[len(missing_trees_memory)] = [point]

    payload = {'missing_trees': []}

    for k, v in missing_trees_memory.items():
        avg_point = np.mean(v, axis=0)
        payload['missing_trees'].append({'lng': avg_point[0], 'lat': avg_point[1]})
    return payload

if __name__ == "__main__":
    missing_trees_orchard(216269, "Bearer <token>")