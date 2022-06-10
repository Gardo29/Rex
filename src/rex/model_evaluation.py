from __future__ import annotations

USER_ID = 0
ITEM_ID = 1
WEIGHT = 2


def precision_k(model,
                test_interactions,
                train_interactions=None,
                k=10,
                threshold: str | int | float = 'auto',
                mode='user'):
    predictions = model.predict(test_interactions, k=k, previous_interactions=train_interactions, mode=mode)

    predictions_pairs = [[column, predicted_id] if mode == 'user' else [predicted_id, column]
                         for column in predictions
                         for predicted_id in predictions[column]]

    filter_condition = [row in predictions_pairs for row in list(test_interactions.iloc[:, :WEIGHT].values)]
    real_recommended = test_interactions[filter_condition]
    id_groups = real_recommended.groupby(real_recommended.columns[USER_ID if mode == 'user' else ITEM_ID])
    
    return {id_group: len(group[group.iloc[:, WEIGHT] >= threshold]) / len(group)
            for id_group, group in id_groups}
