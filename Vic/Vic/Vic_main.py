from typing import TypeAlias,Dict,Any,List

from .Vic_class import VIC,VIC_async
base64str : TypeAlias = str

def Vic(query: str,
        image : list[base64str]|None=None,
        vic : List[str]|None=None,
        extract_info : str|None=None,
        vic_m : bool=False,
        only_vic : bool=False,
        only_vic_with_visual : bool=False,
        visual_all : bool=False,) -> Dict[str,Any]:
    
    # extract_info,vic_m,only_vic,visual_all could not be used simultaneously
    if sum([extract_info is not None,vic_m,only_vic,only_vic_with_visual,visual_all]) > 1:
        raise ValueError("extract_info,vic_m,only_vic,only_vic_with_visual,visual_all could not be used simultaneously")
    if image is None and only_vic is False:
        raise ValueError("Image is required for VIC")
    
    VIC_instance = VIC(query,image)
    
    if vic_m and vic:
        return VIC_instance.vic_m(vic)
    
    if extract_info:
        return VIC_instance.extract_info(extract_info,vic)
    
    if vic:
        return VIC_instance.vic(vic)
    
    
    
    if vic_m:
        return VIC_instance.vic_m_from_scratch()
    
    if only_vic:
        return VIC_instance.only_vic()
    
    if only_vic_with_visual:
        return VIC_instance.only_vic_with_visual()
    
    if visual_all:
        return VIC_instance.visual_all()
    
    return VIC_instance.normal_vic()

async def Vic_async(query: str,
        image : list[base64str],
        vic : List[str]|None=None,
        extract_info : str|None=None,
        vic_m : bool=False,
        only_vic : bool=False,
        only_vic_with_visual : bool=False,
        visual_all : bool=False,) -> Dict[str,Any]:
    
    if sum([extract_info is not None,vic_m,only_vic,visual_all]) > 1:
        raise ValueError("extract_info,vic_m,only_vic,visual_all could not be used simultaneously")
    
    if image is None and only_vic is False:
        raise ValueError("Image is required for VIC")
    
    VIC_instance = VIC_async(query,image)
    
    
    
    if vic_m and vic:
        return await VIC_instance.vic_m(vic)
    
    if vic:
        return await VIC_instance.vic(vic)
    
    if extract_info:
        return await VIC_instance.extract_info(extract_info)
    
    if vic_m:
        return await VIC_instance.vic_m_from_scratch()
    
    if only_vic:
        return await VIC_instance.only_vic()
    
    if only_vic_with_visual:
        return await VIC_instance.only_vic_with_visual()
    
    if visual_all:
        return await VIC_instance.visual_all()
    
    return await VIC_instance.normal_vic()