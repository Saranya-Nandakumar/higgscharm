import numpy as np

def add_scalevar_weight(events, weights_container, variation="nominal"):
    """
    Twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    __doc__:
    ['LHE scale variation weights (w_var / w_nominal)',
    ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
    ' [1] is renscfact=0.5d0 facscfact=1d0 ',
    ' [2] is renscfact=0.5d0 facscfact=2d0 ',
    ' [3] is renscfact=1d0 facscfact=0.5d0 ',
    ' [4] is renscfact=1d0 facscfact=1d0 ',
    ' [5] is renscfact=1d0 facscfact=2d0 ',
    ' [6] is renscfact=2d0 facscfact=0.5d0 ',
    ' [7] is renscfact=2d0 facscfact=1d0 ',
    ' [8] is renscfact=2d0 facscfact=2d0 ']
    """
    lhe_weights = events.LHEScaleWeight
    nom = np.ones(len(weights_container.weight()))
    if variation == "nominal":
        if len(lhe_weights) > 0:
            if len(lhe_weights[0]) == 9:
                nom = lhe_weights[:, 4]
                # index [1] = renscfact=0.5 (muR DOWN), index [7] = renscfact=2.0
                # (muR UP) per the docstring's index table above -- weightUp/weightDown
                # were previously passed as (index1, index7), i.e. swapped relative to
                # Weights.add()'s (name, weight, weightUp, weightDown) signature. Fixed
                # 2026-08-28: weightUp must be the scale-UP variation (index 7).
                weights_container.add(
                    "scalevar_muR",
                    nom,
                    lhe_weights[:, 7] / nom,
                    lhe_weights[:, 1] / nom,
                )
                # Same swap, same fix: index [3] = facscfact=0.5 (muF DOWN),
                # index [5] = facscfact=2.0 (muF UP).
                weights_container.add(
                    "scalevar_muF",
                    nom,
                    lhe_weights[:, 5] / nom,
                    lhe_weights[:, 3] / nom,
                )
                # Same swap: index [0] = both scales DOWN (0.5,0.5), index [8] = both
                # scales UP (2.0,2.0) -- weightUp/weightDown were passed as (index0,
                # index8), i.e. swapped. Also fixed the missing /nom normalization:
                # every other variation here is a ratio to the central point (index 4);
                # this one passed the raw absolute LHE weight instead of the ratio.
                weights_container.add(
                    "scalevar_muR_muF", nom, lhe_weights[:, 8] / nom, lhe_weights[:, 0] / nom
                )
            elif len(lhe_weights[0]) > 1:
                print("Scale variation vector has length ", len(lhe_weights[0]))
        else:
            warnings.warn(
                "LHE scale variation weights are not available, put nominal weights"
            )
            weights_container.add("scalevar_muR", nom, nom, nom)
            weights_container.add("scalevar_muF", nom, nom, nom)
            weights_container.add("scalevar_muR_muF", nom, nom, nom)

    else:
        weights_container.add("scalevar_3pt", nom)