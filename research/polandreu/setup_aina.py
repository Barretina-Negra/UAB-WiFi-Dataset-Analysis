#!/usr/bin/env python3
"""
Setup Script - Projecte Aina Models
UAB THE HACK! 2025
Barcelona Supercomputing Center (BSC)

Aquest script us ajudarà a configurar i provar els models d'IA del Projecte Aina
"""

import sys
import os

# Fix encoding per Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import json


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(number, text):
    """Print step number"""
    print(f"\n[{number}/5] {text}")


def check_dependencies():
    """Check and install required dependencies"""
    print_step(1, "Comprovant dependències...")

    try:
        import requests
        print("✓ requests ja instal·lat")
        return True
    except ImportError:
        print("⚠ requests no trobat. Instal·lant...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "requests"])
            print("✓ requests instal·lat correctament")
            return True
        except Exception as e:
            print(f"✗ Error instal·lant requests: {e}")
            print("\nIntenta manualment: pip install requests")
            return False


def get_api_key():
    """Get or save API key"""
    print_step(2, "Configuració de l'API Key...")

    config_file = ".aina_config.json"

    # Check if config exists
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key', '')
                if api_key:
                    print(f"✓ API Key trobada: {api_key[:10]}...")
                    use_existing = input(
                        "  Vols utilitzar aquesta API key? (s/n): ").lower()
                    if use_existing == 's':
                        return api_key
        except:
            pass

    # Ask for new API key
    print("\n📋 Per obtenir una API key:")
    print("   1. Registra't a https://platform.publicai.co")
    print("   2. Crea un compte gratuït")
    print("   3. Genera una API key des del dashboard")
    print("   4. Copia i enganxa la key aquí")
    print("\n   Documentació: https://platform.publicai.co/docs")
    print("   Ajuda: Pregunta als mentors del BSC")
    api_key = input("\n  Introdueix la teva API Key de PublicAI: ").strip()

    if not api_key:
        print("⚠ No s'ha introduït cap API key")
        return None

    # Save config
    try:
        with open(config_file, 'w') as f:
            json.dump({'api_key': api_key}, f)
        print(f"✓ API Key guardada a {config_file}")
    except Exception as e:
        print(f"⚠ No s'ha pogut guardar l'API key: {e}")

    return api_key


def test_connection(api_key):
    """Test API connection"""
    print_step(3, "Provant connexió amb PublicAI...")

    import requests

    url = "https://api.publicai.co/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UAB-THE-HACK/1.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            models = response.json()["data"]
            print(f"✓ Connexió establerta correctament!")
            print(f"  Models disponibles: {len(models)}")
            return True
        elif response.status_code == 401:
            print("✗ Error d'autenticació. Verifica la teva API key.")
            return False
        else:
            print(f"⚠ Error {response.status_code}: {response.text[:100]}")
            return False

    except requests.exceptions.Timeout:
        print("✗ Timeout. Comprova la teva connexió a internet.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_models(api_key):
    """Test both BSC models"""
    print_step(4, "Provant els models BSC...")

    import requests
    import time

    url = "https://api.publicai.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UAB-THE-HACK/1.0"
    }

    models = {
        "Salamandra-7B": "BSC-LT/salamandra-7b-instruct-tools-16k",
        "ALIA-40B": "BSC-LT/ALIA-40b-instruct_Q8_0"
    }

    for name, model_id in models.items():
        print(f"\n  Provant {name}...")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Hola! Escriu 'OK' si em pots llegir."}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }

        try:
            start = time.time()
            response = requests.post(
                url, headers=headers, json=payload, timeout=30)
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"  ✓ {name}: OK ({elapsed:.2f}s)")
                print(f"    Resposta: {content[:50]}...")
            else:
                print(f"  ✗ {name}: Error {response.status_code}")
                print(f"    {response.text[:100]}")

        except Exception as e:
            print(f"  ✗ {name}: {str(e)[:80]}")

    return True


def create_example_script(api_key):
    """Create an example script for participants"""
    print_step(5, "Creant script d'exemple...")

    example_file = "exemple_aina.py"

    example_code = f'''#!/usr/bin/env python3
"""
Exemple d'ús dels models Projecte Aina
UAB THE HACK! 2025
"""

import requests

# Configuració
API_KEY = "{api_key}"
API_URL = "https://api.publicai.co/v1/chat/completions"

headers = {{
    "Authorization": f"Bearer {{API_KEY}}",
    "Content-Type": "application/json",
    "User-Agent": "UAB-THE-HACK/1.0"
}}

def preguntar(missatge, model="salamandra"):
    """
    Fa una pregunta al model Aina

    Args:
        missatge: La teva pregunta o instrucció
        model: "salamandra" (ràpid) o "alia" (avançat)
    """

    # Seleccionar model
    if model == "alia":
        model_id = "BSC-LT/ALIA-40b-instruct_Q8_0"
    else:
        model_id = "BSC-LT/salamandra-7b-instruct-tools-16k"

    payload = {{
        "model": model_id,
        "messages": [
            {{"role": "system", "content": "Ets un assistent útil en català."}},
            {{"role": "user", "content": missatge}}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {{response.status_code}}: {{response.text}}"

    except Exception as e:
        return f"Error: {{str(e)}}"


# Exemples d'ús
if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLE 1: Pregunta simple amb Salamandra-7B")
    print("=" * 60)

    resposta = preguntar("Què és la intel·ligència artificial?", model="salamandra")
    print(resposta)

    print("\\n" + "=" * 60)
    print("EXEMPLE 2: Generació de codi")
    print("=" * 60)

    resposta = preguntar("Escriu una funció Python per calcular el factorial", model="salamandra")
    print(resposta)

    print("\\n" + "=" * 60)
    print("EXEMPLE 3: Raonament complex amb ALIA-40B")
    print("=" * 60)

    resposta = preguntar("Explica la diferència entre algoritmes greedy i programació dinàmica", model="alia")
    print(resposta)

    print("\\n" + "=" * 60)
    print("Ara és el teu torn! Modifica aquest codi i crea alguna cosa genial!")
    print("=" * 60)
'''

    try:
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_code)
        print(f"✓ Script d'exemple creat: {example_file}")
        print(f"\n  Executa'l amb: python {example_file}")
        return True
    except Exception as e:
        print(f"⚠ No s'ha pogut crear l'exemple: {e}")
        return False


def main():
    """Main setup flow"""
    print_header("🚀 SETUP PROJECTE AINA - UAB THE HACK! 2025")
    print("\nAquest script configurarà el teu entorn per utilitzar els models BSC.")
    print("Trigarà aproximadament 2-3 minuts.")

    input("\nPrem ENTER per començar...")

    # Step 1: Dependencies
    if not check_dependencies():
        print("\n❌ Error instal·lant dependències. Atura't i demana ajuda.")
        return

    # Step 2: API Key
    api_key = get_api_key()
    if not api_key:
        print("\n❌ No es pot continuar sense API key.")
        return

    # Step 3: Test connection
    if not test_connection(api_key):
        print("\n⚠ Hi ha problemes amb la connexió. Demana ajuda als mentors.")
        retry = input("Vols tornar a intentar-ho? (s/n): ").lower()
        if retry == 's':
            if not test_connection(api_key):
                print("\n❌ No s'ha pogut establir connexió.")
                return
        else:
            return

    # Step 4: Test models
    test_models(api_key)

    # Step 5: Create example
    create_example_script(api_key)

    # Success
    print_header("✅ SETUP COMPLETAT!")
    print("\n🎉 Tot llest! Ja pots començar a desenvolupar.")
    print("\n📚 Recursos disponibles:")
    print("   - exemple_aina.py: Script d'exemple bàsic")
    print("   - GUIA_RAPIDA_PARTICIPANTES.md: Documentació completa")
    print("   - test_publicai.py: Script de test avançat")
    print("\n🤝 Si tens problemes:")
    print("   - Demana ajuda als mentors del BSC")
    print("   - Consulta la documentació offline")
    print("\n🚀 Molta sort amb el hackathon!")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup cancel·lat per l'usuari.")
    except Exception as e:
        print(f"\n\n❌ Error inesperat: {e}")
        print("Demana ajuda als mentors!")
