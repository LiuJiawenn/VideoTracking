import xml.etree.ElementTree as ET
from xml.dom import minidom


class XMLAnalyse:
    """This class is used to read and store the annotation info with xml format
    writeXML: write annotation info into a xml file
    analyseXML: for annotations from labelImg and writeXML
    analyseCVATXML: for annotations from CVAT """

    @staticmethod
    def analyseXML(filename):
        suffix_name = filename.split('.').pop()
        if suffix_name == 'xml':
            tree = ET.parse(filename)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)

        return [int(xmin), int(xmax), int(ymin), int(ymax)]

    @staticmethod
    def analyseCVATXML(filename):
        suffix_name = filename.split('.').pop()
        bndbox_list = []
        if suffix_name == 'xml':
            tree = ET.parse(filename)
            root = tree.getroot()
            for box in root.iter('box'):
                xmin = float(box.attrib.get('xtl'))
                ymin = float(box.attrib.get('ytl'))
                xmax = float(box.attrib.get('xbr'))
                ymax = float(box.attrib.get('ybr'))
                bndbox = [int(xmin), int(xmax), int(ymin), int(ymax)]
                bndbox_list.append(bndbox.copy())

        return bndbox_list

    @staticmethod
    def writeXML(address, bndbox):
        # Creating dom tree objects
        doc = minidom.Document()
        # Create root node
        root_node = doc.createElement("annotation")
        doc.appendChild(root_node)

        obj_node = doc.createElement("object")

        bndbox_node = doc.createElement("bndbox")
        for item, value in zip(["xmin", "ymin", "xmax", "ymax"], [bndbox[0], bndbox[2], bndbox[1], bndbox[3]]):
            elem = doc.createElement(item)
            elem.appendChild(doc.createTextNode(str(value)))
            bndbox_node.appendChild(elem)
        obj_node.appendChild(bndbox_node)
        root_node.appendChild(obj_node)

        with open(address, "w", encoding="utf-8") as f:
            # writexml() Para1: the target file
            #            Para2: Indentation format of the root node
            #            Para3: Indentation format of other child nodes
            #            Para4: Line feed format
            #            Para5: Encoding of xml content
            doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")
