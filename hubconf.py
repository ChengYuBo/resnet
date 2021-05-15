#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 March 2019


def MGN_v1(pretrained=False, *args, **kwargs):
    """
    Xception v1 model
    """

    from models.classification.MGN import MGN_v1 as _MGN_v1

    model = _MGN_v1(pretrained=False, *args, **kwargs)

    return model
