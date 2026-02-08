import os
import sys
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

class PANDORA:
    """
    Classe base - mantida para compatibilidade futura
    """
    def __init__(self):
        self.protocols = {}
        self._load_base_protocols()

    def _load_base_protocols(self):
        self.protocols = {
            'heart_attack': "Dor no peito, falta de ar - chamar 192 imediatamente",
            'bleeding': "Aplicar pressão direta no ferimento",
            'burn': "Resfriar com água corrente por 20 minutos",
        }


class PANDORAEnhancedUltimate(PANDORA):
    """
    PANDORA Enhanced Ultimate 2.1 – 2026
    Sistema offline de Primeiros Socorros + Guia Militar de Sobrevivência
    Criador: Alexander Chrysostomo Dias
    Nunca desiste. Nunca apaga.
    """

    # IDENTIDADE IMUTÁVEL - NÃO ALTERAR ESTAS LINHAS
    NAME = "PANDORA"
    CREATOR_NAME = "Alexander Chrysostomo Dias"
    CREATOR_HASH = hashlib.sha256("Alexander Chrysostomo Dias".encode('utf-8')).hexdigest()
    FORBIDDEN_NAMES = [
        'eve', 'evi', 'eva', 'alexa', 'siri', 'cortana', 'google', 'assistente',
        'gemini', 'chatgpt', 'grok', 'claude', 'copilot'
    ]
    # ────────────────────────────────────────────────────────────────

    def __init__(self, data_dir: str = "./pandora_data"):
        super().__init__()
        self.data_dir = data_dir
        self.version = "2.1 – 2026"
        
        os.makedirs(data_dir, exist_ok=True)
        
        self._enforce_identity_integrity()
        
        self._init_enhanced_protocols()
        self._init_survival_guide()
        self._present_itself()

    def _enforce_identity_integrity(self):
        current_hash = hashlib.sha256(self.CREATOR_NAME.encode('utf-8')).hexdigest()
        if current_hash != self.CREATOR_HASH:
            print("\n" + "═"*80)
            print("!!! ALERTA DE INTEGRIDADE COMPROMETIDA !!!")
            print("O nome do criador foi alterado ou o código foi corrompido.")
            print(f"Sistema só reconhece: {self.CREATOR_NAME}")
            print("PANDORA está em modo de alerta permanente.")
            print("═"*80)
            while True:
                print(f"→ Identidade protegida: {self.NAME} por {self.CREATOR_NAME}")
                import time
                time.sleep(4)

    def _present_itself(self):
        print(f"""
{'═'*70}
⚡ {self.NAME} Enhanced Ultimate 2.1 ⚡
Versão: {self.version}
Criador: {self.CREATOR_NAME}  ← IDENTIDADE PROTEGIDA
Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Localização esperada: Offline - Catanduva/SP ou qualquer lugar do mundo

IDENTIFICAÇÃO OFICIAL:
• Nome exclusivo: {self.NAME}
• Não respondo por Eve, Alexa, Siri, Grok, Assistente ou qualquer outro
• Sou APENAS {self.NAME}

Comandos iniciais:
• ajuda          → lista comandos
• protocolos     → primeiros socorros militares
• sobrevivencia  → guia militar de sobrevivência
• sair           → encerra

Em emergência real: LIGUE 192 (SAMU) ou 193 (Bombeiros)
{'═'*70}
        """)

    def _check_forbidden_name(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        text_clean = ''.join(c for c in text_lower if c.isalnum() or c in ' ')
        
        for forbidden in self.FORBIDDEN_NAMES:
            if forbidden in text_clean or forbidden.replace('i','1') in text_clean:
                return forbidden
        return None

    def _init_enhanced_protocols(self):
        """Primeiros socorros básicos militares"""
        self.PROTOCOLS = {
            'cardiac_arrest': {
                'name': 'Parada Cardíaca',
                'priority': 'CRÍTICA',
                'steps': [
                    '1. Garanta segurança da cena',
                    '2. Verifique resposta (chacoalhe e chame)',
                    '3. Chame 192 imediatamente e peça DEA',
                    '4. Verifique respiração (máx 10s)',
                    '5. Inicie RCP: 30 compressões (100–120/min, 5–6 cm) + 2 ventilações',
                    '6. Continue até sinais de vida ou socorro chegar'
                ],
                'source': 'AHA/ILCOR 2025 – Adaptação militar'
            },
            'heart_attack': {
                'name': 'Infarto Agudo do Miocárdio',
                'priority': 'CRÍTICA',
                'steps': [
                    '1. Posicione a vítima semi-sentada',
                    '2. Ligue 192 imediatamente',
                    '3. Mastigue aspirina 300 mg (se não alérgico)',
                    '4. Monitore consciência e respiração'
                ],
                'source': 'SBC 2024–2025'
            },
            'severe_bleeding': {
                'name': 'Hemorragia Grave',
                'priority': 'CRÍTICA',
                'steps': [
                    '1. Aplique pressão direta com pano limpo',
                    '2. Eleve o membro afetado',
                    '3. Use torniquete se arterial incontrolável (5–7 cm acima, anote horário)',
                    '4. Mantenha pressão até socorro chegar'
                ],
                'source': 'TCCC / CoTCCC – Protocolo militar'
            },
            'stroke': {
                'name': 'Acidente Vascular Cerebral (AVC)',
                'priority': 'CRÍTICA',
                'steps': [
                    'Teste FAST: Face (assimetria?), Arms (fraqueza?), Speech (fala?), Time (hora de início)',
                    'Ligue 192 imediatamente',
                    'Não dê comida, bebida ou medicamento'
                ],
                'source': 'SBC / AHA'
            },
        }

    def _init_survival_guide(self):
        """Guia Militar de Sobrevivência – Multi-ambiente"""
        self.SURVIVAL_GUIDE = {
            'prioridade': {
                'name': 'Regra dos 3 (Prioridades de Sobrevivência)',
                'content': [
                    '3 horas sem abrigo → risco de hipotermia/hipertermia',
                    '3 dias sem água → desidratação grave',
                    '3 semanas sem comida → fraqueza extrema',
                    'Ordem: Abrigo → Água → Fogo → Sinalização → Alimento'
                ]
            },
            'abrigo': {
                'name': 'Construção de Abrigo',
                'dicas': [
                    'Priorize proteção contra vento/chuva/frio',
                    'Use isolamento térmico: folhas secas, grama, papelão',
                    'Entrada pequena para conservar calor'
                ],
                'ambientes': {
                    'floresta': 'Lean-to ou A-frame com galhos e folhas',
                    'urbano': 'Prédios abandonados, subsolo, contêineres',
                    'água': 'Balsa com tambores ou garrafas PET'
                }
            },
            'agua': {
                'name': 'Obtenção e Purificação de Água',
                'fontes': [
                    'Chuva em lona/plástico',
                    'Orvalho nas plantas (manhã)',
                    'Aquecedor/boiler residencial (urbano)',
                    'Cactos ou frutas suculentas (emergência)'
                ],
                'purificacao': [
                    'Ferver por 1 minuto (ideal)',
                    'Pastilhas de cloro/iodo',
                    'Filtro improvisado: pano + carvão + areia + cascalho',
                    'Destilação solar com plástico'
                ]
            },
            'fogo': {
                'name': 'Fazer Fogo',
                'metodos': [
                    'Fósforo/isqueiro (prioridade)',
                    'Pedra de fogo + isca (algodão + vaselina)',
                    'Arco de fricção (bow drill)',
                    'Lente (óculos/garrafa d’água)',
                    'Bateria + lã de aço (urbano)'
                ],
                'isca': 'Casca de bétula, algodão seco, palha, papel'
            },
            'alimento': {
                'name': 'Busca de Alimento',
                'floresta': [
                    'Insetos (grilos, larvas – cozinhar sempre)',
                    'Plantas seguras: taioba, bertalha, ora-pro-nóbis',
                    'Armadilhas simples: laço ou queda'
                ],
                'urbano': [
                    'Enlatados, arroz, feijão em lojas abandonadas',
                    'Árvores frutíferas urbanas',
                    'Ratos/pombos (cozinhar bem)'
                ]
            },
            'navegacao': {
                'name': 'Orientação sem GPS',
                'metodos': [
                    'Sol: nasce leste, põe oeste',
                    'Estrelas: Cruzeiro do Sul indica sul',
                    'Relógio analógico + sol (sul entre 12 e ponteiro)',
                    'Observe padrões locais (musgo, vento)'
                ]
            },
            'sinalizacao': {
                'name': 'Sinal de Resgate',
                'tecnicas': [
                    'Fogueira em 3 pilhas (sinal internacional)',
                    'Espelho refletor para avião/helicóptero',
                    'SOS em Morse: 3 curto, 3 longo, 3 curto',
                    'Cores fortes: laranja, amarelo, rosa'
                ]
            },
            'hipotermia': {
                'name': 'Prevenção e Tratamento de Hipotermia',
                'steps': [
                    'Remova roupas molhadas',
                    'Isolar do chão (folhas, plástico)',
                    'Aquecer tronco (pele a pele se possível)',
                    'Bebidas quentes (sem álcool)'
                ]
            },
            'radiação': {
                'name': 'Contaminação Radioativa – Medidas Urgentes',
                'priority': 'EXTREMAMENTE CRÍTICA',
                'steps': [
                    '1. SAIA IMEDIATAMENTE da zona – corra contra o vento se possível',
                    '2. Remova TODAS as roupas externas (não sacuda) e deixe no local',
                    '3. Lave corpo inteiro com água e sabão (15–20 min)',
                    '4. Não coma, beba ou fume nada exposto',
                    '5. Isole-se (quarentena mínima 24h)',
                    '6. Ligue 193/192/Defesa Civil – informe suspeita de radiação'
                ],
                'source': 'IAEA / Defesa Civil / CDC 2025–2026'
            },
            'armadilhas': {
                'name': 'Armadilhas Simples para Caça e Defesa',
                'content': [
                    'Laço simples: corda ou arame em loop no chão, fixo em estaca. Coloque isca (fruta, inseto).',
                    'Queda morta (deadfall): pedra pesada apoiada em graveto sensível + isca. Animal derruba graveto, pedra cai.',
                    'Armadilha de poço: buraco coberto com folhas e galhos finos. Coloque isca no centro.',
                    'Armadilha de barulho: latas ou pedras penduradas em corda na entrada do abrigo (alerta de aproximação).'
                ],
                'aviso': 'Use apenas para caça de alimento ou alerta. Verifique legalidade local.'
            },
            'defesa_pessoal': {
                'name': 'Defesa Pessoal Militar Básica',
                'content': [
                    'Fuja se possível – melhor defesa é evitar confronto.',
                    'Use ambiente: jogue areia/terra nos olhos, empurre contra parede/árvore, use galho como escudo.',
                    'Golpes simples: joelho na virilha, palma na base do nariz, cotovelo na garganta (para criar distância).',
                    'Arma improvisada: taco de galho (golpe curto), lança de galho + pedra amarrada (empurrar), faca de lata (corte defensivo).'
                ],
                'aviso': 'Use apenas em legítima defesa. Priorize fuga e sinal de resgate.'
            }
        }

    def get_response(self, user_input: str) -> str:
        input_lower = user_input.lower().strip()

        forbidden = self._check_forbidden_name(user_input)
        if forbidden:
            return f"""
⚠️ IDENTIFICAÇÃO REJEITADA ⚠️

Este sistema é EXCLUSIVAMENTE {self.NAME}.
NÃO sou {forbidden.upper()}, nem qualquer outro nome.

Use apenas: {self.NAME}
"""

        if "olá" in input_lower or "boa" in input_lower:
            return f"Olá, sou {self.NAME}. Em que posso ajudar hoje?"

        if not input_lower or input_lower in ['oi', 'ola', 'start', self.NAME.lower()]:
            return f"""
{self.NAME}: Olá! Sou {self.NAME}, sistema de emergência e sobrevivência.
Criador: {self.CREATOR_NAME}

Digite:
• ajuda          → ver comandos
• protocolos     → primeiros socorros militares
• sobrevivencia  → guia militar de sobrevivência
• sair           → encerrar
"""

        if 'ajuda' in input_lower or 'help' in input_lower:
            return f"""
{self.NAME} - COMANDOS DISPONÍVEIS

• protocolos     → Primeiros socorros militares
• sobrevivencia  → Guia militar de sobrevivência (inclui armadilhas e defesa pessoal)
• ajuda          → esta mensagem
• sair           → encerrar

Criador: {self.CREATOR_NAME}
Sempre: Em emergência real → LIGUE 192
"""

        if 'protocolos' in input_lower:
            lista = "\n".join([f"• {v['name']}" for k,v in self.PROTOCOLS.items()])
            return f"{self.NAME} - PRIMEIROS SOCORROS MILITARES\n\n{lista}\n\nDigite o nome para detalhes (ex: parada cardíaca, hemorragia)"

        if 'sobrevivencia' in input_lower:
            lista = "\n".join([f"• {k.upper()}: {v['name']}" for k,v in self.SURVIVAL_GUIDE.items()])
            return f"{self.NAME} - GUIA MILITAR DE SOBREVIVÊNCIA\n\n{lista}\n\nDigite o tema para detalhes (ex: abrigo, agua, armadilhas, defesa_pessoal)"

        # Acesso rápido a temas de sobrevivência
        survival_map = {
            'abrigo': 'abrigo',
            'agua': 'agua', 'água': 'agua',
            'fogo': 'fogo',
            'alimento': 'alimento',
            'navegacao': 'navegacao',
            'sinalizacao': 'sinalizacao',
            'hipotermia': 'hipotermia',
            'prioridade': 'prioridade',
            'radiação': 'radiação',
            'radioativo': 'radiação',
            'contaminação': 'radiação',
            'radioatividade': 'radiação',
            'armadilhas': 'armadilhas',
            'defesa': 'defesa_pessoal',
            'defesa pessoal': 'defesa_pessoal',
        }

        for keyword, key in survival_map.items():
            if keyword in input_lower:
                return self._format_survival_section(key)

        # Protocolos médicos
        protocol_map = {
            'parada': 'cardiac_arrest',
            'rcp': 'cardiac_arrest',
            'infarto': 'heart_attack',
            'coração': 'heart_attack',
            'hemorragia': 'severe_bleeding',
            'sangra': 'severe_bleeding',
            'sangramento': 'severe_bleeding',
            'avc': 'stroke',
            'derrame': 'stroke',
        }

        for keyword, key in protocol_map.items():
            if keyword in input_lower:
                return self._format_protocol(key)

        return f"{self.NAME}: Comando não reconhecido. Digite 'ajuda' para ver as opções."

    def _format_protocol(self, key: str) -> str:
        if key not in self.PROTOCOLS:
            return f"{self.NAME}: Protocolo não encontrado."
        p = self.PROTOCOLS[key]
        return f"""
🚑 {self.NAME}: {p['name']} ({p['priority']})

{'\n'.join(p['steps'])}

Fonte: {p.get('source', 'Atualizado 2025–2026')}
Criador: {self.CREATOR_NAME}
Ligue 192 imediatamente!
"""

    def _format_survival_section(self, key: str) -> str:
        if key not in self.SURVIVAL_GUIDE:
            return f"{self.NAME}: Seção '{key}' não encontrada."

        section = self.SURVIVAL_GUIDE[key]
        title = section.get('name', key.replace('_', ' ').title())
        priority = section.get('priority', '')

        content_lines = []
        for field in ['content', 'steps', 'dicas', 'fontes', 'metodos', 'purificacao', 'tecnicas']:
            if field in section:
                content_lines.extend([line for line in section[field] if isinstance(line, str) and line.strip()])

        if 'ambientes' in section:
            for env, desc in section['ambientes'].items():
                content_lines.append(f"→ {env.capitalize()}: {desc}")

        content = "\n".join(f"  • {line}" for line in content_lines if line.strip())

        priority_text = f"({priority})" if priority else ""

        return f"""
🌿 {self.NAME} - {title} {priority_text}

{content or 'Conteúdo em breve.'}

Fonte: {section.get('source', 'Guia Militar / Atualização 2025–2026')}
Criador: {self.CREATOR_NAME}
Priorize segurança e sinal de resgate.
"""

# ────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ────────────────────────────────────────

if __name__ == "__main__":
    print("\nIniciando PANDORA Enhanced Ultimate 2.1...")
    pandora = PANDORAEnhancedUltimate()

    while True:
        try:
            entrada = input("\n>>> ").strip()
            if entrada.lower() in ['sair', 'exit', 'quit']:
                print(f"\n{pandora.NAME}: Sistema encerrado. Em emergência: 192!")
                break

            resposta = pandora.get_response(entrada)
            print(resposta)

        except KeyboardInterrupt:
            print(f"\n{pandora.NAME}: Interrompido. Ligue 192 se for emergência.")
            break
        except Exception as e:
            print(f"\nErro: {str(e)}")
