from metrics.gbc import get_gbc_score
from metrics.hscore import get_hscore, get_regularized_h_score
from metrics.jcnce import get_jcnce_score
from metrics.logme import LogME
from metrics.leep import get_leep_score
from metrics.nleep import get_nleep_score
from metrics.emd import get_emd_score
from metrics.nce import get_nce_score
from metrics.transrate import get_transrate_score
from metrics.ids import get_ids_score
from metrics.rsa_dds import get_rsa_score, get_dds_score, get_parc_score



def evaluate_predicted_transferability(
    metric: str,
    pca_dim: int, 
    **kwargs
):
    score = 0.0
    if metric == 'emd':
        score = get_emd_score(features_src=kwargs['features_src'], features_tar=kwargs['features_tar'], labels_src=kwargs['labels_src'], labels_tar=kwargs['labels_tar'], device=kwargs['device'])
    elif metric == 'easy_emd':
        score = get_emd_score(features_src=kwargs['features_src'], features_tar=kwargs['features_tar'], device=kwargs['device'], easy=True)
    elif metric == 'ids':
        score = get_ids_score(features_src=kwargs['features_src'], features_tar=kwargs['features_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'gbc':
        score = get_gbc_score(features=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'hscore':
        score = get_hscore(features=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'shhscore':
        score = get_regularized_h_score(features=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'jcnce':
        score = get_jcnce_score(features_src=kwargs['features_src'], features_tar=kwargs['features_tar'], labels_src=kwargs['labels_src'], labels_tar=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'logme':
        score = LogME().fit(f=kwargs['features_tar'].cpu().detach().numpy(), y=kwargs['labels_tar'].cpu().detach().numpy(), pca_dim=pca_dim)
    elif metric == 'nleep':
        score = get_nleep_score(features=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'transrate':
        score = get_transrate_score(features=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'], pca_dim=pca_dim)
    elif metric == 'leep':
        score = get_leep_score(logits=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'])
    elif metric == 'nce':
        score = get_nce_score(logits=kwargs['features_tar'], target_labels=kwargs['labels_tar'], device=kwargs['device'])
    elif metric == 'rsa':
        score = get_rsa_score(x=kwargs['features_src'], y=kwargs['features_tar'], dist=kwargs['dist'], feature_norm=kwargs['feature_norm'])
    elif metric == 'dds':
        score = get_dds_score(x=kwargs['features_src'], y=kwargs['features_tar'], dist=kwargs['dist'], feature_norm=kwargs['feature_norm'])
    elif metric == 'parc':
        score = get_parc_score(x=kwargs['features_tar'], labels=kwargs['labels_tar'])
    else:
        print(f"Metric {metric} is not supported yet!")
    
    return score